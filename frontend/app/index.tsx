// app/index.tsx
import React, { useEffect, useState } from 'react';
import { SafeAreaView, ActivityIndicator, View, Text } from 'react-native';
import * as SecureStore from 'expo-secure-store';
import { useAuthStore } from '@/store/authStore';
import { refreshTokens } from '@/api/auth/auth';
import { getProfile } from '@/api/user/user';
import { useRouter } from 'expo-router';
import { Button } from 'react-native-paper';

export default function Index() {
    const [loading, setLoading] = useState(true);
    const { setUser, updateAccessToken, clearUser, user } = useAuthStore();
    const router = useRouter();

    useEffect(() => {
        const initAuth = async () => {
            try {
                // Retrieve the current access token from SecureStore.
                let token = await SecureStore.getItemAsync('accessToken');

                // If a token exists, attempt to refresh it.
                if (token) {
                    const refreshRes = await refreshTokens();
                    if (refreshRes && refreshRes.accessToken) {
                        token = refreshRes.accessToken;
                        // Update SecureStore with the new token.
                        await SecureStore.setItemAsync('accessToken', token as string, {
                            keychainAccessible: SecureStore.AFTER_FIRST_UNLOCK,
                        });
                        updateAccessToken(token as string);
                    }
                }

                // If we have a valid token, fetch the user's profile.
                if (token) {
                    const profileData = await getProfile();
                    if (profileData && profileData.user) {
                        setUser(profileData.user, token);
                    }
                }
            } catch (err) {
                console.error('Auth initialization error:', err);
                clearUser();
            } finally {
                setLoading(false);
            }
        };

        initAuth();
    }, [setUser, updateAccessToken, clearUser]);

    useEffect(() => {
        // Once initialization is complete and a user is set, redirect to the main tabs.
        if (!loading && user) {
            router.replace('/(tabs)');
        }
    }, [loading, user, router]);

    if (loading) {
        return (
            <SafeAreaView style={{ flex: 1, backgroundColor: '#1F2937', justifyContent: 'center', alignItems: 'center' }}>
                <ActivityIndicator size="large" color="#fff" />
            </SafeAreaView>
        );
    }

    // If not loading and no user is found, prompt for login.
    return (
        <SafeAreaView style={{ flex: 1, backgroundColor: '#1F2937' }}>
            <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
                <Text style={{ color: '#fff', fontSize: 24 }}>Welcome to the App! Please log in.</Text>
                <Button mode="contained" onPress={() => router.push('/auth/Login')}>Login</Button>
            </View>
        </SafeAreaView>
    );
}
