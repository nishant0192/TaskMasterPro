import React from 'react';
import { SafeAreaView, FlatList, View } from 'react-native';
import CustomText from '@/components/CustomText';
import CustomButton from '@/components/CustomButton';
import { useGetTasks, useCompleteTask, useDeleteTask } from '@/api/task/useTask';
import { useRouter } from 'expo-router';
import { MaterialIcons } from '@expo/vector-icons';
import { Colors } from '@/constants/Colors';

type Task = {
    id: string;
    title: string;
    description?: string;
    status: string;
    createdAt: string;
};

const TaskItem: React.FC<{ task: Task }> = ({ task }) => {
    const router = useRouter();
    const completeMutation = useCompleteTask();
    const deleteMutation = useDeleteTask();

    const handleComplete = () => {
        completeMutation.mutate({ id: task.id });
    };

    const handleDelete = () => {
        deleteMutation.mutate({ id: task.id });
    };

    return (
        <View
            className="p-4 rounded mb-2"
            style={{
                backgroundColor: Colors.SECONDARY_BACKGROUND, // Dark Charcoal
                borderWidth: 1,
                borderColor: Colors.DIVIDER,
            }}
        >
            <CustomText variant="headingMedium" className="mb-1" style={{ color: Colors.PRIMARY_TEXT }}>
                {task.title}
            </CustomText>
            {task.description && (
                <CustomText variant="headingSmall" className="mb-1" style={{ color: Colors.SECONDARY_TEXT }}>
                    {task.description}
                </CustomText>
            )}
            <View className="flex-row justify-between">
                <CustomButton
                    title="Details"
                    onPress={() =>
                        router.push({
                            pathname: "/task/TaskDetailsScreen",
                            params: { id: task.id },
                        })
                    }
                    className="py-2 px-4 rounded"
                    style={{ backgroundColor: Colors.ACCENT }} // Electric Pink
                    textStyle={{ color: Colors.PRIMARY_TEXT }}
                />
                {task.status !== 'DONE' && (
                    <CustomButton
                        title="Complete"
                        onPress={handleComplete}
                        className="py-2 px-4 rounded"
                        style={{ backgroundColor: Colors.SUCCESS }} // Green
                        textStyle={{ color: Colors.PRIMARY_TEXT }}
                    />
                )}
                <CustomButton
                    title="Delete"
                    onPress={handleDelete}
                    className="py-2 px-4 rounded"
                    style={{ backgroundColor: Colors.ERROR }} // Red
                    textStyle={{ color: Colors.PRIMARY_TEXT }}
                />
            </View>
        </View>
    );
};

export default function TaskListScreen() {
    const { data, isLoading, error } = useGetTasks();
    const router = useRouter();

    if (isLoading) {
        return (
            <SafeAreaView
                className="flex-1 justify-center items-center"
                style={{ backgroundColor: Colors.PRIMARY_BACKGROUND }}
            >
                <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT }}>
                    Loading tasks...
                </CustomText>
            </SafeAreaView>
        );
    }

    if (error) {
        return (
            <SafeAreaView
                className="flex-1 justify-center items-center"
                style={{ backgroundColor: Colors.PRIMARY_BACKGROUND }}
            >
                <CustomText variant="headingMedium" style={{ color: Colors.ERROR }}>
                    Error: {error.message}
                </CustomText>
            </SafeAreaView>
        );
    }

    return (
        <SafeAreaView
            style={{
                flex: 1,
                backgroundColor: Colors.PRIMARY_BACKGROUND,
                position: 'relative', // Ensure children absolute positioning is relative to this container.
            }}
        >
            <FlatList
                data={data.tasks}
                keyExtractor={(item: Task) => item.id}
                renderItem={({ item }) => <TaskItem task={item} />}
                contentContainerStyle={{ padding: 16, paddingBottom: 120 }} // Extra bottom space for sticky button
            />
            <View
                className="absolute bottom-4"
                style={{
                    position: 'absolute',
                    bottom: 0,
                    width: '100%',
                    borderTopWidth: 1,
                    borderTopColor: Colors.DIVIDER,
                    paddingVertical: 30,
                    paddingHorizontal: 24,
                }}
            >
                <CustomButton
                    title="Create Task"
                    onPress={() => router.push('/task/CreateTaskScreen')}
                    style={{
                        backgroundColor: Colors.BUTTON, // Vibrant Orange
                        paddingVertical: 12,
                        borderRadius: 8,
                    }}
                    textStyle={{
                        color: Colors.PRIMARY_TEXT, // Light Gray
                        textAlign: 'center',
                        fontWeight: 'bold',
                        fontSize: 20,
                    }}
                />
            </View>
        </SafeAreaView>
    );
}
