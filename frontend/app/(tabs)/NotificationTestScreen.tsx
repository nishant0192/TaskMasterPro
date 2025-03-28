import React, { useEffect } from 'react';
import { SafeAreaView, Button, Text, Alert } from 'react-native';
import { useRegisterPushToken, useTestNotification } from '@/api/notifications/useNotifications';
import * as Notifications from 'expo-notifications';

export default function NotificationTestScreen() {
  const { mutate: registerToken } = useRegisterPushToken();
  const { mutate: testNotification } = useTestNotification();

  useEffect(() => {
    async function init() {
      const { status } = await Notifications.requestPermissionsAsync();  // Request notification permissions
      if (status === 'granted') {
        try {
          const tokenResponse = await Notifications.getExpoPushTokenAsync();  // Get Expo push token
          const token = tokenResponse.data;
          console.log('Expo Device Push Token:', token);  // Log Expo token (e.g., ExponentPushToken[xxxxxxxxxxx])

          // Send the Expo push token to the backend
          if (token) {
            registerToken({ expoPushToken: token });  // Send Expo push token to backend
          }
        } catch (error) {
          console.error('Error fetching device push token:', error);
        }
      } else {
        console.log('Push notification permissions not granted');
      }
    }

    init();

    // Foreground notification handler
    const foregroundSubscription = Notifications.addNotificationReceivedListener(notification => {
      console.log('Notification received in foreground:', notification);
      // Display an in-app notification, e.g., alert or custom UI notification
      Alert.alert(notification.request.content.title ?? 'Notification', notification.request.content.body ?? 'New notification received');
    });

    // Background notification handler (when tapped or opened)
    const backgroundSubscription = Notifications.addNotificationResponseReceivedListener(response => {
      console.log('Notification tapped or opened:', response);
      // Handle the background notification response (e.g., navigate to a specific screen)
    });

    // Clean up listeners on component unmount
    return () => {
      foregroundSubscription.remove();
      backgroundSubscription.remove();
    };
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
