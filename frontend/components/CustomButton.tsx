import React, { useRef } from 'react';
import { TouchableWithoutFeedback, Animated, StyleProp, ViewStyle, TextStyle, View } from 'react-native';
import CustomText from './CustomText';

type CustomButtonProps = {
  title: string;
  onPress: () => void;
  style?: StyleProp<ViewStyle>;
  textStyle?: StyleProp<TextStyle>;
  className?: string;
  activeScale?: number;
  icon?: React.ReactNode;
  disabled?: boolean;
};

const CustomButton: React.FC<CustomButtonProps> = ({
  title,
  onPress,
  style,
  textStyle,
  className,
  activeScale = 0.95,
  icon,
  disabled = false,
}) => {
  const scaleAnim = useRef(new Animated.Value(1)).current;

  const handlePressIn = () => {
    if (disabled) return;
    Animated.spring(scaleAnim, {
      toValue: activeScale,
      useNativeDriver: true,
    }).start();
  };

  const handlePressOut = () => {
    if (disabled) return;
    Animated.spring(scaleAnim, {
      toValue: 1,
      friction: 3,
      useNativeDriver: true,
    }).start(() => onPress());
  };

  return (
    <TouchableWithoutFeedback onPressIn={handlePressIn} onPressOut={handlePressOut}>
      <Animated.View
        style={[
          {
            transform: [{ scale: scaleAnim }],
            flexDirection: 'row',
            alignItems: 'center',
            justifyContent: 'center',
            opacity: disabled ? 0.5 : 1,
          },
          style,
        ]}
        className={className}
      >
        {icon && <View style={{ marginRight: 8 }}>{icon}</View>}
        <CustomText variant="headingMedium" style={textStyle}>
          {title}
        </CustomText>
      </Animated.View>
    </TouchableWithoutFeedback>
  );
};

export default CustomButton;
