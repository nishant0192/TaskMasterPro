import React, { useEffect, useState } from 'react';
import { Text, View, Button } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { useAuthStore } from '@/store/authStore';
import * as SecureStore from 'expo-secure-store';
import * as Notifications from 'expo-notifications';

export default function HomeScreen() {
  const router = useRouter();
  const user = useAuthStore((state) => state.user);
  const [accessToken, setAccessToken] = useState<string | null>(null);

  useEffect(() => {
    // Request notification permissions when the component mounts
    (async () => {
      const { status } = await Notifications.requestPermissionsAsync();
      if (status !== 'granted') {
        console.log('Notification permissions not granted');
      } else {
        console.log('Notification permissions granted');
      }
    })();

    // Retrieve accessToken from SecureStore (async)
    (async () => {
      const token = await SecureStore.getItemAsync("accessToken");
      setAccessToken(token);
    })();
  }, []);

  return (
    <SafeAreaView className="flex-1 p-4">
      <View className="flex-1 justify-center items-center">
        <Text className="text-xl mb-4 text-white">
          Welcome, {accessToken || 'Loading token...'}
        </Text>
        {user ? (
          <>
            <Text className="text-xl mb-4 text-white">Welcome, {user.email}</Text>
            <Text className="text-xl mb-4 text-white">User ID: {user.id}</Text>
            <Text className="text-xl mb-4 text-white">Name: {user.name}</Text>
            <Button title="Go to Dashboard" onPress={() => router.push('/task/TaskListScreen')} />
          </>
        ) : (
          <Button title="Click to Login" onPress={() => router.push('/auth/Login')} />
        )}
      </View>
    </SafeAreaView>
  );
}
