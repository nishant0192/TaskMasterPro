import React, { useState } from 'react';
import { SafeAreaView, View } from 'react-native';
import { useRouter } from 'expo-router';
import CustomInput from '@/components/CustomInput';
import CustomText from '@/components/CustomText';
import CustomButton from '@/components/CustomButton';
import { useSignin } from '@/api/auth/useAuth';

const Login = () => {
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

  return (
    <SafeAreaView className="flex-1 bg-white px-4 py-8">
      <View className="flex-1">
        <CustomText variant="pageHeader" className="text-center mb-6">
          Login
        </CustomText>
        <View className="mb-4">
          <CustomText variant="headingSmall" className="mb-2">
            Email
          </CustomText>
          <CustomInput
            value={email}
            onChangeText={setEmail}
            placeholder="Enter your email"
            keyboardType="email-address"
            autoCapitalize="none"
            className="border border-gray-300 rounded p-2"
          />
        </View>
        <View className="mb-6">
          <CustomText variant="headingSmall" className="mb-2">
            Password
          </CustomText>
          <CustomInput
            value={password}
            onChangeText={setPassword}
            placeholder="Enter your password"
            secureTextEntry
            className="border border-gray-300 rounded p-2"
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
          className="bg-blue-500 py-3 rounded mb-4"
          textStyle={{ color: 'white', textAlign: 'center', fontWeight: 'bold' }}
        />
        <CustomButton
          title="Don't have an account? Register"
          onPress={() => router.push('/auth/register')}
          className="bg-transparent py-3"
          textStyle={{ color: 'blue', textAlign: 'center' }}
        />
      </View>
    </SafeAreaView>
  );
};

export default Login;
