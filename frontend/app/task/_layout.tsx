import React from 'react';
import { Stack } from 'expo-router';
import Colors from '@/constants/Colors'; // Ensure the path is correct
import { TasksProvider } from '@/context/TasksContext';

export default function TaskLayout() {
    return (
        <TasksProvider>

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
                    options={{ title: 'Tasks List' }}
                />
                <Stack.Screen
                    name="CreateTaskScreen"
                    options={{ title: 'Create Task' }}
                />
                <Stack.Screen
                    name="TaskDetailsScreen"
                    options={{ title: 'Task Details' }}
                />
                <Stack.Screen
                    name="ArchivedTaskListScreen"
                    options={{ title: 'Archived Tasks' }}
                />
            </Stack>
        </TasksProvider>
    );
}
