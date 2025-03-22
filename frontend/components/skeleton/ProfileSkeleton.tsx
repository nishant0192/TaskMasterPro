import React, { useRef, useEffect } from 'react';
import { View, Animated } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

const ProfileSkeleton = () => {
  const shimmerAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.loop(
      Animated.timing(shimmerAnim, {
        toValue: 1,
        duration: 1000,
        useNativeDriver: true,
      })
    ).start();
  }, [shimmerAnim]);

  const shimmerStyle = {
    opacity: shimmerAnim.interpolate({
      inputRange: [0, 0.5, 1],
      outputRange: [0.3, 1, 0.3],
    }),
  };

  return (
    <SafeAreaView className="flex-col items-center space-y-4">
      {/* Circle for avatar */}
      <Animated.View
        style={[
          { width: 100, height: 100, borderRadius: 50, backgroundColor: '#E5E7EB', marginBottom: 16 },
          shimmerStyle,
        ]}
      />
      {/* Rectangle for name */}
      <Animated.View
        style={[
          { width: '70%', height: 32, borderRadius: 4, backgroundColor: '#E5E7EB', marginBottom: 16 },
          shimmerStyle,
        ]}
      />
      {/* Rectangle for email */}
      <Animated.View
        style={[
          { width: '50%', height: 32, borderRadius: 4, backgroundColor: '#E5E7EB', marginBottom: 16 },
          shimmerStyle,
        ]}
      />
      {/* Rectangle for last login */}
      <Animated.View
        style={[
          { width: '50%', height: 32, borderRadius: 4, backgroundColor: '#E5E7EB' },
          shimmerStyle,
        ]}
      />
    </SafeAreaView>
  );
};

export default ProfileSkeleton;
