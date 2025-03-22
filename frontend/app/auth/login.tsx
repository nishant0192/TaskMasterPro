import React, { useState } from 'react';
import { SafeAreaView, TextInput, TouchableOpacity, Text, View } from 'react-native';
import { useSignin } from '@/api/auth/useAuth';
import { useRouter } from 'expo-router';

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  // Get the entire mutation object
  const signinMutation = useSignin();
  const { mutate: signin, error, status } = signinMutation;
  const isLoading = status === 'pending';

  const router = useRouter();

  const handleLogin = () => {
    signin(
      { email, password },
      {
        onSuccess: (data) => {
          console.log('Login successful', data);
          // Navigate to home or dashboard after successful login
          router.push('/');
        },
        onError: (err) => {
          console.error('Error signing in', err);
        },
      }
    );
  };

  return (
    <SafeAreaView className="flex-1 bg-white px-4 py-8">
      <Text className="text-3xl font-bold text-center mb-6">Login</Text>
      <View className="mb-4">
        <Text className="text-base mb-2">Email</Text>
        <TextInput
          value={email}
          onChangeText={setEmail}
          placeholder="Enter your email"
          keyboardType="email-address"
          autoCapitalize="none"
          className="border border-gray-300 rounded p-2"
        />
      </View>
      <View className="mb-6">
        <Text className="text-base mb-2">Password</Text>
        <TextInput
          value={password}
          onChangeText={setPassword}
          placeholder="Enter your password"
          secureTextEntry
          className="border border-gray-300 rounded p-2"
        />
      </View>
      {error && (
        <Text className="text-red-500 text-center mb-4">{error.message}</Text>
      )}
      <TouchableOpacity
        onPress={handleLogin}
        disabled={isLoading}
        className="bg-blue-500 py-3 rounded mb-4"
      >
        <Text className="text-white text-center font-bold">
          {isLoading ? 'Loading...' : 'Login'}
        </Text>
      </TouchableOpacity>
      <TouchableOpacity onPress={() => router.push('/auth/register')}>
        <Text className="text-blue-500 text-center">
          Don't have an account? Register
        </Text>
      </TouchableOpacity>
    </SafeAreaView>
  );
};

export default Login;
