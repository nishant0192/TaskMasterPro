import React, { useState } from 'react';
import { SafeAreaView, View, Alert } from 'react-native';
import { useRouter } from 'expo-router';
import CustomText from '@/components/CustomText';
import CustomButton from '@/components/CustomButton';
import StyledInput from '@/components/StyledInput';
import { useSignup } from '@/api/auth/useAuth';

export default function Register() {
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
          router.push('/auth/Login');
        },
        onError: (err) => {
          console.error('Registration error', err);
          Alert.alert("Error", "Registration failed, please try again.");
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
        <CustomText variant="pageHeader" className="text-center mb-8 text-white">
          Register
        </CustomText>
        <View className="mb-4">
          <CustomText variant="headingSmall" className="mb-2 text-gray-300">
            Name
          </CustomText>
          <StyledInput
            mode="outlined"
            labelText="Name"
            value={name}
            onChangeText={setName}
            placeholder="Enter your name"
            keyboardType="default"
            colors={inputColors}
          />
        </View>
        <View className="mb-4">
          <CustomText variant="headingSmall" className="mb-2 text-gray-300">
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
        <View className="mb-4">
          <CustomText variant="headingSmall" className="mb-2 text-gray-300">
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
        <View className="mb-6">
          <CustomText variant="headingSmall" className="mb-2 text-gray-300">
            Confirm Password
          </CustomText>
          <StyledInput
            mode="outlined"
            labelText="Confirm Password"
            value={confirmPassword}
            onChangeText={setConfirmPassword}
            placeholder="Confirm your password"
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
          title={isMutating ? 'Registering...' : 'Register'}
          onPress={handleRegister}
          className="bg-green-600 py-3 rounded mb-4 shadow-lg"
          textStyle={{ color: 'white', textAlign: 'center', fontWeight: 'bold' }}
        />
        <CustomButton
          title="Already have an account? Login"
          onPress={() => router.push('/auth/Login')}
          className="bg-transparent py-3"
          textStyle={{ color: 'lightblue', textAlign: 'center' }}
        />
      </View>
    </SafeAreaView>
  );
}
