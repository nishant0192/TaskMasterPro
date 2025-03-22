// frontend/components/CustomInput.tsx

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
    outputRange: ['gray', 'blue'],
  });

  return (
    <Animated.View
      style={[
        containerStyle,
        { borderColor, borderWidth: 1, borderRadius: 4, padding: 4 },
      ]}
      className={className}
    >
      <TextInput
        {...props}
        onFocus={handleFocus}
        onBlur={handleBlur}
        style={[{ padding: 10 }, style]}
      />
    </Animated.View>
  );
};

export default CustomInput;
