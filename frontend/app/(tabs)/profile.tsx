import React, { useState, useEffect } from 'react';
import { SafeAreaView, View, Image } from 'react-native';
import CustomText from '@/components/CustomText';
import ProfileSkeleton from '@/components/skeleton/ProfileSkeleton';
import { useGetProfile } from '@/api/user/useUser';
import { useAuthStore } from '@/store/authStore';
import { MaterialIcons } from '@expo/vector-icons';

export default function ProfileScreen() {
    const [minLoading, setMinLoading] = useState(true);
    const userInStore = useAuthStore((state) => state.user);

    // Enable the query only if the user is not already in the store.
    const { data, isLoading, error } = useGetProfile({ enabled: !userInStore });

    // Ensure the skeleton is displayed for at least 3 seconds.
    useEffect(() => {
        const timer = setTimeout(() => {
            setMinLoading(false);
        }, 3000);
        return () => clearTimeout(timer);
    }, []);

    // Use the user from the store if available, otherwise use fetched data.
    const user = userInStore || (data && data.user);

    if (isLoading || minLoading) {
        return (
            <SafeAreaView className="flex-1 bg-white p-4">
                <ProfileSkeleton />
            </SafeAreaView>
        );
    }

    if (error || !user) {
        return (
            <SafeAreaView className="flex-1 bg-white p-4">
                <CustomText variant="headingMedium" className="text-red-500 text-center">
                    Error fetching profile: {error ? error.message : 'User not found'}
                </CustomText>
            </SafeAreaView>
        );
    }

    return (
        <SafeAreaView className="flex-1 bg-white p-4">
            <View className="items-center">
                {user.profileImage ? (
                    <Image source={{ uri: user.profileImage }} className="w-24 h-24 rounded-full" />
                ) : (
                    <MaterialIcons name="account-circle" size={96} color="#ccc" />
                )}
                <CustomText variant="pageHeader" className="mt-4">
                    {user.name || 'Unnamed User'}
                </CustomText>
                <CustomText variant="headingMedium" className="mt-2">
                    {user.email}
                </CustomText>
                <CustomText variant="headingSmall" className="mt-2">
                    Last Login: {new Date(user.lastLogin).toLocaleString()}
                </CustomText>
            </View>
        </SafeAreaView>
    );
}
