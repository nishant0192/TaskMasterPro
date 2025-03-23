import React from 'react';
import { Stack } from 'expo-router';

export default function AuthLayout() {
    return (
        <Stack>
            <Stack.Screen
                name="Login"
                options={{
                    title: 'Login',
                }}
            />
            <Stack.Screen
                name="Register"
                options={{
                    title: 'Create Account',
                }}
            />
        </Stack>
    );
}
