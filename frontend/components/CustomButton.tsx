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
};

const CustomButton: React.FC<CustomButtonProps> = ({
  title,
  onPress,
  style,
  textStyle,
  className,
  activeScale = 0.95,
  icon,
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
        style={[{ transform: [{ scale: scaleAnim }], flexDirection: 'row', alignItems: 'center', justifyContent: 'center' }, style]}
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
