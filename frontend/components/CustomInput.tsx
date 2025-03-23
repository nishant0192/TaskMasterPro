import React, { useState, useRef } from 'react';
import { TextInput, Animated, StyleProp, ViewStyle, TextInputProps } from 'react-native';

type CustomInputProps = TextInputProps & {
  containerStyle?: StyleProp<ViewStyle>;
  className?: string;
};

const CustomInput: React.FC<CustomInputProps> = ({ containerStyle, className, style, ...props }) => {
  const [isFocused, setIsFocused] = useState(false);
  const borderAnim = useRef(new Animated.Value(0)).current;

  const handleFocus = () => {
    setIsFocused(true);
    Animated.timing(borderAnim, {
      toValue: 1,
      duration: 200,
      useNativeDriver: false,
    }).start();
  };

  const handleBlur = () => {
    setIsFocused(false);
    Animated.timing(borderAnim, {
      toValue: 0,
      duration: 200,
      useNativeDriver: false,
    }).start();
  };

  const borderColor = borderAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ['#555', '#1E90FF'], // Gray to DodgerBlue
  });

  return (
    <Animated.View
      style={[
        containerStyle,
        { borderColor, borderWidth: 1, borderRadius: 8, padding: 8 },
      ]}
      className={className}
    >
      <TextInput
        {...props}
        onFocus={handleFocus}
        onBlur={handleBlur}
        style={[
          { padding: 12, color: '#fff', backgroundColor: '#333' },
          style,
        ]}
        placeholderTextColor="#888"
      />
    </Animated.View>
  );
};

export default CustomInput;
