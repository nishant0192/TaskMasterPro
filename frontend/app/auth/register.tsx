import React, { useState } from 'react';
import { SafeAreaView, View, Alert } from 'react-native';
import { useRouter } from 'expo-router';
import CustomInput from '@/components/CustomInput';
import CustomText from '@/components/CustomText';
import CustomButton from '@/components/CustomButton';
import { useSignup } from '@/api/auth/useAuth';

const Register = () => {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const router = useRouter();
  const { mutate: signup, error, status } = useSignup();
  const isMutating = status === 'pending';

  const handleRegister = () => {
    if (password !== confirmPassword) {
      Alert.alert("Error", "Passwords don't match");
      return;
    }
    signup(
      { name, email, password },
      {
        onSuccess: (data) => {
          console.log('Registration successful', data);
          router.push('/auth/login');
        },
        onError: (err) => {
          console.error('Registration error', err);
          Alert.alert("Error", "Registration failed, please try again.");
        },
      }
    );
  };

  return (
    <SafeAreaView className="flex-1 bg-white px-4 py-8">
      <View className="flex-1">
        <CustomText variant="pageHeader" className="text-center mb-6">
          Register
        </CustomText>
        <View className="mb-4">
          <CustomText variant="headingSmall" className="mb-2">
            Name
          </CustomText>
          <CustomInput
            value={name}
            onChangeText={setName}
            placeholder="Enter your name"
            className="border border-gray-300 rounded p-2"
          />
        </View>
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
        <View className="mb-4">
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
        <View className="mb-6">
          <CustomText variant="headingSmall" className="mb-2">
            Confirm Password
          </CustomText>
          <CustomInput
            value={confirmPassword}
            onChangeText={setConfirmPassword}
            placeholder="Confirm your password"
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
          title={isMutating ? 'Registering...' : 'Register'}
          onPress={handleRegister}
          className="bg-green-500 py-3 rounded mb-4"
          textStyle={{ color: 'white', textAlign: 'center', fontWeight: 'bold' }}
        />
        <CustomButton
          title="Already have an account? Login"
          onPress={() => router.push('/auth/login')}
          className="bg-transparent py-3"
          textStyle={{ color: 'blue', textAlign: 'center' }}
        />
      </View>
    </SafeAreaView>
  );
};

export default Register;
