import React, { useState, useRef, useEffect } from "react";
import {
  Keyboard,
  KeyboardTypeOptions,
  StyleProp,
  TextStyle,
  View,
  NativeSyntheticEvent,
  TextInputContentSizeChangeEventData,
} from "react-native";
import { TextInput } from "react-native-paper";
import { AntDesign, EvilIcons } from "@expo/vector-icons";
import CustomText from "@/components/CustomText";
import { Colors } from "@/constants/Colors";

export interface StyledInputProps extends React.ComponentProps<typeof TextInput> {
  mode: "outlined" | "flat";
  labelText?: string;
  value: string;
  autoFocus?: boolean;
  onChangeText: (text: string) => void;
  trackingData?: Record<string, any>;
  success?: boolean | null; // true for success, false for error, null for neutral
  placeholder?: string;
  colors?: {
    backgroundColor?: string;
    textColor?: string;
    placeholderTextColor?: string;
    neutralBorderColor?: string;
    successBorderColor?: string;
    errorBorderColor?: string;
    clearIconColor?: string;
    labelColor?: string;
  };
  style?: StyleProp<TextStyle>;
  multiline?: boolean;
  numberOfLines?: number;
  autoCapitalize?: "none" | "sentences" | "words" | "characters";
  keyboardType?: KeyboardTypeOptions;
  maxLength?: number;
}

const StyledInput: React.FC<StyledInputProps> = ({
  mode,
  labelText,
  value,
  autoFocus = false,
  onChangeText,
  trackingData = {},
  success = null,
  placeholder,
  colors = {},
  style,
  multiline = false,
  numberOfLines = 1,
  autoCapitalize = "none",
  keyboardType,
  maxLength,
  ...props
}) => {
  // Initialize local state with the prop value.
  const [inputValue, setInputValue] = useState(value);

  // Update the local state when the value prop changes.
  useEffect(() => {
    setInputValue(value);
  }, [value]);

  const fixedHeight = multiline ? 120 : 50;
  const inputRef = useRef<any>(null);

  const minHeight = 50;
  const maxHeight = 200;

  // Default colors
  const {
    backgroundColor = Colors.SECONDARY_BACKGROUND,
    textColor = Colors.PRIMARY_TEXT,
    placeholderTextColor = Colors.SECONDARY_TEXT,
    neutralBorderColor = Colors.DIVIDER,
    successBorderColor = Colors.SUCCESS,
    errorBorderColor = Colors.ERROR,
    clearIconColor = Colors.INACTIVE,
  } = colors;

  const handleChange = (text: string) => {
    let processedText = text;

    if (keyboardType === "numeric") {
      processedText = processedText.replace(/[^0-9]/g, "");
      if (processedText) {
        const num = parseInt(processedText, 10);
        if (num < 1) processedText = "1";
        if (num > 5) processedText = "5";
      }
    }

    setInputValue(processedText);
    onChangeText(processedText);
  };

  const handleClear = () => {
    setInputValue("");
    onChangeText("");
  };

  const handleBlur = () => {
    const trimmed = inputValue.trim();
    if (trimmed !== inputValue) {
      setInputValue(trimmed);
      onChangeText(trimmed);
    }
  };

  const borderColor =
    success === true ? successBorderColor : success === false ? errorBorderColor : neutralBorderColor;

  useEffect(() => {
    const keyboardDidHideListener = Keyboard.addListener("keyboardDidHide", () => {
      inputRef.current?.blur();
    });
    return () => {
      keyboardDidHideListener.remove();
    };
  }, []);

  const effectiveMaxLength = multiline ? maxLength ?? 150 : maxLength;

  const handleContentSizeChange = (
    event: NativeSyntheticEvent<TextInputContentSizeChangeEventData>
  ) => {
    if (multiline) {
      const { height } = event.nativeEvent.contentSize;
      const newHeight = Math.max(minHeight, Math.min(height, maxHeight));
      setInputValue(inputValue); // Force update if necessary
    }
  };

  return (
    <View>
      <TextInput
        ref={inputRef}
        clearButtonMode="never"
        mode={mode}
        autoFocus={autoFocus}
        cursorColor={textColor}
        outlineStyle={{ borderRadius: 10, borderColor }}
        textColor={textColor}
        style={[
          {
            backgroundColor,
            fontFamily: "Poppins_400Regular",
            marginVertical: 8,
            shadowColor: "rgba(0, 0, 0, 0.2)",
            shadowOffset: { width: 0, height: 2 },
            shadowOpacity: 0.2,
            shadowRadius: 10,
            elevation: 5,
            height: multiline ? fixedHeight : 50,
            textAlignVertical: multiline ? "top" : "center",
          },
          style,
        ]}
        label={labelText}
        placeholder={placeholder}
        contentStyle={{ color: textColor, fontFamily: "Poppins_400Regular" }}
        placeholderTextColor={placeholderTextColor}
        value={inputValue}
        onChangeText={handleChange}
        onContentSizeChange={handleContentSizeChange}
        onBlur={handleBlur}
        keyboardType={keyboardType}
        editable={success !== true}
        onFocus={() => {
          if (success === false) {
            handleClear();
          }
        }}
        multiline={multiline}
        numberOfLines={numberOfLines}
        autoCapitalize={autoCapitalize}
        maxLength={effectiveMaxLength}
        {...props}
        theme={{
          colors: {
            primary: textColor,
            background: backgroundColor,
            surface: backgroundColor,
            error: errorBorderColor,
            onSurfaceVariant: placeholderTextColor,
          },
          fonts: {
            labelSmall: {
              fontFamily: "Poppins_400Regular",
            },
          },
        }}
        right={
          inputValue && success == null ? (
            <TextInput.Icon
              icon={() => <AntDesign name="closecircle" size={24} color={clearIconColor} />}
              onPress={handleClear}
            />
          ) : success ? (
            <TextInput.Icon
              icon={() => <EvilIcons name="check" size={24} color={Colors.SUCCESS} />}
            />
          ) : null
        }
      />
      {multiline && (
        <View style={{ alignItems: "flex-end", marginTop: 4, marginRight: 8 }}>
          <CustomText style={{ color: placeholderTextColor }} variant="headingSmall">
            {inputValue.length}/{effectiveMaxLength}
          </CustomText>
        </View>
      )}
    </View>
  );
};

export default StyledInput;
