// frontend/components/CustomText.tsx

import React, { useRef, useEffect } from 'react';
import { Animated, StyleProp, TextStyle } from 'react-native';
import { Text } from 'react-native';

type CustomTextProps = {
  children: React.ReactNode;
  style?: StyleProp<TextStyle>;
  className?: string;
};

const CustomText: React.FC<CustomTextProps> = ({ children, style, className }) => {
  const fadeAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(fadeAnim, {
      toValue: 1,
      duration: 500,
      useNativeDriver: true,
    }).start();
  }, [fadeAnim]);

  return (
    <Animated.Text style={[style, { opacity: fadeAnim }]} className={className}>
      {children}
    </Animated.Text>
  );
};

export default CustomText;
