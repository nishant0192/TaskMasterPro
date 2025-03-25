import React, { useEffect } from 'react';
import { SafeAreaView, Button, Text } from 'react-native';
import { useRegisterPushToken, useTestNotification } from '@/api/notifications/useNotifications';
import * as Notifications from 'expo-notifications';

export default function NotificationTestScreen() {
  const { mutate: registerToken } = useRegisterPushToken();
  const { mutate: testNotification } = useTestNotification();

  useEffect(() => {
    async function init() {
      const { status } = await Notifications.requestPermissionsAsync();
      if (status === 'granted') {
        try {
          const tokenResponse = await Notifications.getDevicePushTokenAsync();
          const token = tokenResponse.data;
          console.log('FCM Device Push Token:', token);
          // Register token on backend
          registerToken({ expoPushToken: token });
        } catch (error) {
          console.error('Error fetching device push token:', error);
        }
      } else {
        console.log('Push notification permissions not granted');
      }
    }
    init();
  }, [registerToken]);

  return (
    <SafeAreaView style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
      <Text>Notification Test</Text>
      <Button title="Send Test Notification" onPress={async () => {
        try {
          const tokenResponse = await Notifications.getDevicePushTokenAsync();
          const token = tokenResponse.data;
          console.log('FCM Device Push Token:', token);
          if (token) {
            testNotification({ expoPushToken: token });
          }
        } catch (error) {
          console.error('Error fetching device push token:', error);
        }
      }} />
    </SafeAreaView>
  );
}
