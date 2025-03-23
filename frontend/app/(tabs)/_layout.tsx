import { Tabs } from 'expo-router';
import React from 'react';
import { Platform } from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons'; // Import Material Icons

import { HapticTab } from '@/components/HapticTab';
import TabBarBackground from '@/components/ui/TabBarBackground';
import { Colors } from '@/constants/Colors';

export default function TabLayout() {
  // Directly use dark theme colors from the Colors file
  const currentColors = {
    PRIMARY_BACKGROUND: Colors.PRIMARY_BACKGROUND,
    SECONDARY_BACKGROUND: Colors.SECONDARY_BACKGROUND,
    ACCENT: Colors.ACCENT,
    INACTIVE: Colors.INACTIVE,
    PRIMARY_TEXT: Colors.PRIMARY_TEXT,
    SECONDARY_TEXT: Colors.SECONDARY_TEXT,
  };

  return (
    <Tabs
      screenOptions={{
        tabBarActiveTintColor: currentColors.ACCENT,  // Use ACCENT color for active tab
        tabBarInactiveTintColor: currentColors.INACTIVE, // Use INACTIVE color for inactive tabs
        headerShown: false,
        tabBarButton: HapticTab,
        tabBarBackground: TabBarBackground,
        tabBarStyle: Platform.select({
          ios: {
            // Use a transparent background on iOS to show the blur effect
            position: 'absolute',
            backgroundColor: currentColors.SECONDARY_BACKGROUND, // Background color for iOS
          },
          default: {
            backgroundColor: currentColors.SECONDARY_BACKGROUND, // Background color for Android
          },
        }),
      }}>
      <Tabs.Screen
        name="index"
        options={{
          title: 'Home',
          tabBarIcon: ({ color }) => <Icon name="home" size={28} color={color} />, // Home icon
        }}
      />
      <Tabs.Screen
        name="explore"
        options={{
          title: 'Explore',
          tabBarIcon: ({ color }) => <Icon name="explore" size={28} color={color} />, // Explore icon
        }}
      />
      <Tabs.Screen
        name="profile"
        options={{
          title: 'Profile',
          tabBarIcon: ({ color }) => <Icon name="person" size={28} color={color} />, // Profile icon
        }}
      />
    </Tabs>
  );
}
