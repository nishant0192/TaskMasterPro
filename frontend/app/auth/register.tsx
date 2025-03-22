import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, Alert } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { useSignup } from '@/api/auth/useAuth'; // Adjust import as needed

const Register = () => {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const router = useRouter();

  // Get the signup mutation
  const { mutate: signup, status, error } = useSignup();
  const isLoading = status === 'pending';

  const handleRegister = () => {
    if (password !== confirmPassword) {
      Alert.alert("Error", "Passwords don't match");
      return;
    }
    signup(
      { name, email, password },
      {
        onSuccess: (data) => {
          console.log('Signup successful', data);
          // Navigate to login or home after successful registration
          router.push('/auth/login');
        },
        onError: (err) => {
          console.error('Error signing up', err);
          Alert.alert("Error", "Registration failed, please try again.");
        },
      }
    );
  };

  return (
    <SafeAreaView className="flex-1 bg-white px-4 py-8">
      <Text className="text-3xl font-bold text-center mb-6">Register</Text>
      <View className="mb-4">
        <Text className="mb-2 text-base">Name</Text>
        <TextInput
          value={name}
          onChangeText={setName}
          placeholder="Enter your name"
          className="border border-gray-300 rounded p-2"
        />
      </View>
      <View className="mb-4">
        <Text className="mb-2 text-base">Email</Text>
        <TextInput
          value={email}
          onChangeText={setEmail}
          placeholder="Enter your email"
          keyboardType="email-address"
          autoCapitalize="none"
          className="border border-gray-300 rounded p-2"
        />
      </View>
      <View className="mb-4">
        <Text className="mb-2 text-base">Password</Text>
        <TextInput
          value={password}
          onChangeText={setPassword}
          placeholder="Enter your password"
          secureTextEntry
          className="border border-gray-300 rounded p-2"
        />
      </View>
      <View className="mb-6">
        <Text className="mb-2 text-base">Confirm Password</Text>
        <TextInput
          value={confirmPassword}
          onChangeText={setConfirmPassword}
          placeholder="Confirm your password"
          secureTextEntry
          className="border border-gray-300 rounded p-2"
        />
      </View>
      {/* {error && (
        <Text className="text-red-500 text-center mb-4">{error.message}</Text>
      )} */}
      <TouchableOpacity
        onPress={handleRegister}
        disabled={isLoading}
        className="bg-green-500 py-3 rounded mb-4"
      >
        <Text className="text-white text-center font-bold">
          {isLoading ? 'Registering...' : 'Register'}
        </Text>
      </TouchableOpacity>
      <TouchableOpacity onPress={() => router.push('/auth/login')}>
        <Text className="text-blue-500 text-center">
          Already have an account? Login
        </Text>
      </TouchableOpacity>
    </SafeAreaView>
  );
};

export default Register;
