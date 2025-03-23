import React, { useState, useEffect, useRef } from 'react';
import { TextInput, Animated, View, TextInputProps } from 'react-native';

interface FloatingLabelInputProps extends TextInputProps {
  label: string;
  containerStyle?: object;
  inputStyle?: object;
}

export const FloatingLabelInput: React.FC<FloatingLabelInputProps> = ({
  label,
  containerStyle,
  inputStyle,
  ...props
}) => {
  const [isFocused, setIsFocused] = useState(false);
  const animatedIsFocused = useRef(new Animated.Value(props.value ? 1 : 0)).current;

  useEffect(() => {
    Animated.timing(animatedIsFocused, {
      toValue: isFocused || props.value ? 1 : 0,
      duration: 150,
      useNativeDriver: false,
    }).start();
  }, [isFocused, props.value]);

  const labelStyle = {
    position: 'absolute' as const,
    left: 12,
    top: animatedIsFocused.interpolate({
      inputRange: [0, 1],
      outputRange: [18, -8],
    }),
    fontSize: animatedIsFocused.interpolate({
      inputRange: [0, 1],
      outputRange: [16, 12],
    }),
    color: animatedIsFocused.interpolate({
      inputRange: [0, 1],
      outputRange: ['#aaa', '#fff'],
    }),
    backgroundColor: 'transparent',
    paddingHorizontal: 4,
  };

  return (
    <View style={[{ marginTop: 20 }, containerStyle]}>
      <Animated.Text style={labelStyle}>{label}</Animated.Text>
      <TextInput
        {...props}
        style={[
          {
            height: 48,
            color: '#fff',
            backgroundColor: '#333',
            borderWidth: 1,
            borderColor: '#555',
            borderRadius: 8,
            paddingHorizontal: 12,
            paddingTop: 18, // leave space for label
          },
          inputStyle,
        ]}
        placeholderTextColor="#888"
        onFocus={() => setIsFocused(true)}
        onBlur={() => setIsFocused(false)}
      />
    </View>
  );
};

export default FloatingLabelInput;
