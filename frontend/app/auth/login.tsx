import React, { useState } from 'react';
import { SafeAreaView, View } from 'react-native';
import { useRouter } from 'expo-router';
import CustomText from '@/components/CustomText';
import CustomButton from '@/components/CustomButton';
import StyledInput from '@/components/StyledInput';
import { useSignin } from '@/api/auth/useAuth';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const { mutate: signin, error, status } = useSignin();
  const isMutating = status === 'pending';
  const router = useRouter();

  const handleLogin = () => {
    signin(
      { email, password },
      {
        onSuccess: (data) => {
          console.log('Login successful', data);
          router.push('/'); // Navigate to home/dashboard
        },
        onError: (err) => {
          console.error('Error signing in', err);
        },
      }
    );
  };

  // Custom dark-mode color palette for inputs
  const inputColors = {
    backgroundColor: "#1F2937",
    textColor: "#fff",
    placeholderTextColor: "#888",
    neutralBorderColor: "#4B5563",
    successBorderColor: "#32CD32",
    errorBorderColor: "#FF4500",
    clearIconColor: "#9CA3AF",
  };

  return (
    <SafeAreaView className="flex-1 bg-gray-900 p-6">
      <View className="flex-1">
        <View className="mb-4">
          <CustomText variant="headingMedium" className="mb-2 text-gray-300">
            Email
          </CustomText>
          <StyledInput
            mode="outlined"
            labelText="Email"
            value={email}
            onChangeText={setEmail}
            placeholder="Enter your email"
            keyboardType="email-address"
            autoCapitalize="none"
            colors={inputColors}
          />
        </View>
        <View className="mb-6">
          <CustomText variant="headingMedium" className="mb-2 text-gray-300">
            Password
          </CustomText>
          <StyledInput
            mode="outlined"
            labelText="Password"
            value={password}
            onChangeText={setPassword}
            placeholder="Enter your password"
            secureTextEntry
            colors={inputColors}
          />
        </View>
        {error && (
          <CustomText variant="headingSmall" className="text-red-500 text-center mb-4">
            {error.message}
          </CustomText>
        )}
        <CustomButton
          title={isMutating ? 'Loading...' : 'Login'}
          onPress={handleLogin}
          className="bg-blue-500 py-3 rounded mb-4 shadow-lg"
          textStyle={{ color: 'white', textAlign: 'center', fontWeight: 'bold' }}
        />
        <CustomButton
          title="Don't have an account? Register"
          onPress={() => router.push('/auth/Register')}
          className="bg-transparent py-3"
          textStyle={{ color: 'lightblue', textAlign: 'center' }}
        />
      </View>
    </SafeAreaView>
  );
}
