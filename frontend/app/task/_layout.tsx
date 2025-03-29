import React from 'react';
import { Stack } from 'expo-router';
import { router } from 'expo-router';
import Colors from '@/constants/Colors'; // Ensure the path is correct
import { TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons'; // Optional: You can use any icon library

export default function TaskLayout() {

    return (
        <Stack
            screenOptions={{
                headerStyle: { backgroundColor: Colors.SECONDARY_BACKGROUND },
                headerTitleStyle: { color: Colors.PRIMARY_TEXT, fontFamily: "Poppins_700Bold" },
                headerTintColor: Colors.ACCENT,
                headerTitleAlign: 'center',
            }}
        >
            <Stack.Screen
                name="TaskListScreen"
                options={{
                    title: 'Tasks List',
                    headerLeft: () => (
                        <TouchableOpacity onPress={() => router.replace('/')}>
                            <Ionicons name="arrow-back" size={24} color={Colors.ACCENT} />
                        </TouchableOpacity>
                    ),
                }}
            />
            <Stack.Screen
                name="CreateTaskScreen"
                options={{
                    title: 'Create Task',
                    headerLeft: () => (
                        <TouchableOpacity onPress={() => router.replace('/task/TaskListScreen')}>
                            <Ionicons name="arrow-back" size={24} color={Colors.ACCENT} />
                        </TouchableOpacity>
                    ),
                }}
            />
            <Stack.Screen
                name="TaskDetailsScreen"
                options={{
                    title: 'Task Details',
                    headerLeft: () => (
                        <TouchableOpacity onPress={() => router.replace('/task/TaskListScreen')}>
                            <Ionicons name="arrow-back" size={24} color={Colors.ACCENT} />
                        </TouchableOpacity>
                    ),
                }}
            />
            <Stack.Screen
                name="ArchivedTaskListScreen"
                options={{
                    title: 'Archived Tasks',
                    headerBackVisible: false,
                }}
            />
        </Stack>
    );
}
