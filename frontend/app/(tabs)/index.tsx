import React from 'react';
import { Text, View, Button } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { useAuthStore } from '@/store/authStore';
import * as secureStorage from 'expo-secure-store';

export default function HomeScreen() {
  const router = useRouter();
  const user = useAuthStore((state) => state.user);

  return (
    <SafeAreaView className="flex-1 p-4">
      <View className="flex-1 justify-center items-center">
        <Text className="text-xl mb-4 text-white">Welcome, {secureStorage.getItem("accessToken")}</Text>
        {user ? (
          <>
            <Text className="text-xl mb-4 text-white">Welcome, {user.email}</Text>
            <Text className="text-xl mb-4 text-white">Welcome, {user.id}</Text>
            <Text className="text-xl mb-4 text-white">Welcome, {user.name}</Text>
            <Button title="Go to Dashboard" onPress={() => router.push('/task/TaskListScreen')} />
          </>
        ) : (
          // <Button title="Click to Login" onPress={() => router.push('/task/CreateTaskScreen')} />
          <Button title="Click to Login" onPress={() => router.push('/auth/Login')} />
        )}
      </View>
    </SafeAreaView>
  );
}
