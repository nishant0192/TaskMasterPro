import React, { useRef, useEffect } from 'react';
import { Animated, StyleProp, TextStyle } from 'react-native';
import { Text } from 'react-native';
import textStyles from '@/constants/textStyles';

type CustomTextProps = {
  children: React.ReactNode;
  // Variant should be one of the keys defined in textStyles
  variant?: keyof typeof textStyles;
  style?: StyleProp<TextStyle>;
  className?: string;
};

const CustomText: React.FC<CustomTextProps> = ({
  children,
  variant = 'pageHeader',
  style,
  className,
}) => {
  const fadeAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(fadeAnim, {
      toValue: 1,
      duration: 500,
      useNativeDriver: true,
    }).start();
  }, [fadeAnim]);

  // Combine the variant style from textStyles with any additional style passed.
  const combinedStyle: StyleProp<TextStyle> = [textStyles[variant], style];

  return (
    <Animated.Text style={[combinedStyle, { opacity: fadeAnim }]} className={className}>
      {children}
    </Animated.Text>
  );
};

export default CustomText;
