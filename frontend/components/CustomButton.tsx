// frontend/components/CustomButton.tsx

import React, { useRef } from 'react';
import { TouchableWithoutFeedback, Animated, StyleProp, ViewStyle, TextStyle, Text } from 'react-native';
import CustomText from './CustomText';

// You can pass tailwind classes via the "className" prop if you're using nativewind
type CustomButtonProps = {
  title: string;
  onPress: () => void;
  style?: StyleProp<ViewStyle>;
  textStyle?: StyleProp<TextStyle>;
  className?: string;
  activeScale?: number;
};

const CustomButton: React.FC<CustomButtonProps> = ({
  title,
  onPress,
  style,
  textStyle,
  className,
  activeScale = 0.95,
}) => {
  const scaleAnim = useRef(new Animated.Value(1)).current;

  const handlePressIn = () => {
    Animated.spring(scaleAnim, {
      toValue: activeScale,
      useNativeDriver: true,
    }).start();
  };

  const handlePressOut = () => {
    Animated.spring(scaleAnim, {
      toValue: 1,
      friction: 3,
      useNativeDriver: true,
    }).start(() => onPress());
  };

  return (
    <TouchableWithoutFeedback onPressIn={handlePressIn} onPressOut={handlePressOut}>
      <Animated.View
        style={[{ transform: [{ scale: scaleAnim }] }, style]}
        className={className}
      >
        <CustomText variant={"headingMedium"} style={textStyle}>{title}</CustomText>
      </Animated.View>
    </TouchableWithoutFeedback>
  );
};

export default CustomButton;
