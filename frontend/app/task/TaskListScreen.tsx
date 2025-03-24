import React, { useState, useEffect } from 'react';
import { SafeAreaView, FlatList, TouchableOpacity, Modal, View } from 'react-native';
import CustomText from '@/components/CustomText';
import CustomButton from '@/components/CustomButton';
import { useRouter } from 'expo-router';
import { MaterialIcons, Ionicons } from '@expo/vector-icons';
import { Colors } from '@/constants/Colors';
import { useTasksStore } from '@/store/tasksStore';
import { useGetTasks, useUpdateTask, useDeleteTask } from '@/api/task/useTask';
import { Task } from '@/types/tasks';

type TaskItemProps = {
    task: Task;
    isSelected: boolean;
    multiSelect: boolean;
    onToggleSelect: (id: string) => void;
    onActionComplete: () => void;
};

const TaskItem: React.FC<TaskItemProps> = ({
    task,
    isSelected,
    multiSelect,
    onToggleSelect,
    onActionComplete,
}) => {
    const router = useRouter();
    const updateTaskMutation = useUpdateTask();
    const deleteTaskMutation = useDeleteTask();
    const [menuVisible, setMenuVisible] = useState(false);

    // If in multi-select mode, tapping toggles selection; otherwise, navigate to details.
    const handlePress = () => {
        if (multiSelect) {
            onToggleSelect(task.id);
        } else {
            router.push({
                pathname: '/task/TaskDetailsScreen',
                params: { id: task.id },
            });
        }
    };

    // Options for the per-item menu.
    const options = [
        {
            title: 'Details',
            icon: <Ionicons name="information-circle-outline" size={18} color={Colors.PRIMARY_TEXT} />,
            action: () => {
                setMenuVisible(false);
                router.push({
                    pathname: '/task/TaskDetailsScreen',
                    params: { id: task.id },
                });
                onActionComplete();
            },
        },
        ...(task.status !== 'DONE'
            ? [
                {
                    title: 'Complete',
                    icon: <MaterialIcons name="done-all" size={18} color={Colors.PRIMARY_TEXT} />,
                    action: () => {
                        setMenuVisible(false);
                        updateTaskMutation.mutate(
                            { id: task.id, status: 'DONE' },
                            { onSuccess: onActionComplete }
                        );
                    },
                },
            ]
            : []),
        {
            title: 'Delete',
            icon: <MaterialIcons name="delete" size={18} color={Colors.PRIMARY_TEXT} />,
            action: () => {
                setMenuVisible(false);
                deleteTaskMutation.mutate(
                    { id: task.id },
                    { onSuccess: onActionComplete }
                );
            },
        },
        {
            title: 'Archive',
            icon: <MaterialIcons name="archive" size={18} color={Colors.PRIMARY_TEXT} />,
            action: () => {
                setMenuVisible(false);
                updateTaskMutation.mutate(
                    { id: task.id, isArchived: true },
                    { onSuccess: onActionComplete }
                );
            },
        },
    ];

    return (
        <>
            <TouchableOpacity
                activeOpacity={0.7}
                onLongPress={() => onToggleSelect(task.id)}
                onPress={handlePress}
                className={`p-4 rounded mb-2 flex-row items-center ${isSelected ? 'border-2 border-blue-500' : 'border border-gray-600'
                    }`}
                style={{ backgroundColor: Colors.SECONDARY_BACKGROUND }}
            >
                <View className="flex-1">
                    <CustomText variant="headingMedium" className="mb-1 text-white">
                        {task.title}
                    </CustomText>
                    {task.description && (
                        <CustomText variant="headingSmall" className="text-gray-300">
                            {task.description}
                        </CustomText>
                    )}
                </View>
                <TouchableOpacity onPress={() => setMenuVisible(true)} className="ml-2">
                    <MaterialIcons name="more-vert" size={20} color={Colors.PRIMARY_TEXT} />
                </TouchableOpacity>
            </TouchableOpacity>
            <Modal visible={menuVisible} transparent animationType="fade" onRequestClose={() => setMenuVisible(false)}>
                <TouchableOpacity
                    className="flex-1 justify-center items-center bg-black/50"
                    onPress={() => setMenuVisible(false)}
                >
                    <View className="bg-black p-4 rounded w-4/5">
                        {options.map((option, index) => (
                            <TouchableOpacity
                                key={index}
                                onPress={option.action}
                                className="py-2 px-4 rounded bg-indigo-500 mt-2 flex-row items-center"
                            >
                                <View style={{ width: 30, alignItems: 'center' }}>{option.icon}</View>
                                <CustomText variant="headingMedium" className="text-white ml-2">
                                    {option.title}
                                </CustomText>
                            </TouchableOpacity>
                        ))}
                    </View>
                </TouchableOpacity>
            </Modal>
        </>
    );
};

export default function TaskListScreen() {
    const { tasks, setTasks } = useTasksStore();
    const router = useRouter();
    const [selectedTasks, setSelectedTasks] = useState<string[]>([]);
    const [moreModalVisible, setMoreModalVisible] = useState(false);
    const multiSelect = selectedTasks.length > 0;
    const { data, refetch } = useGetTasks();
    const updateTaskMutation = useUpdateTask();
    const deleteTaskMutation = useDeleteTask();

    // Populate Zustand store on initial load.
    useEffect(() => {
        if (data?.tasks) {
            setTasks(data.tasks);
        }
    }, [data, setTasks]);

    // Refresh tasks from API and update store.
    const refreshTasks = async () => {
        const res = await refetch();
        if (res.data?.tasks) {
            setTasks(res.data.tasks);
        }
    };

    const toggleSelectTask = (id: string) => {
        setSelectedTasks((prev) =>
            prev.includes(id) ? prev.filter(taskId => taskId !== id) : [...prev, id]
        );
    };

    // Multi-select actions.
    const completeSelected = async () => {
        selectedTasks.forEach(id => {
            const task = tasks.find(t => t.id === id);
            if (task && task.status !== 'DONE') {
                updateTaskMutation.mutate({ id, status: 'DONE' });
            }
        });
        setSelectedTasks([]);
        setMoreModalVisible(false);
        refreshTasks();
    };

    const deleteSelected = async () => {
        selectedTasks.forEach(id => {
            deleteTaskMutation.mutate({ id });
        });
        setSelectedTasks([]);
        setMoreModalVisible(false);
        refreshTasks();
    };

    const archiveSelected = async () => {
        selectedTasks.forEach(id => {
            updateTaskMutation.mutate({ id, isArchived: true });
        });
        setSelectedTasks([]);
        setMoreModalVisible(false);
        refreshTasks();
    };

    return (
        <SafeAreaView className="flex-1 bg-[#121212] relative">
            {/* Header: Navigate to Archived Tasks */}
            <TouchableOpacity
                onPress={() => router.push('/task/ArchivedTaskListScreen')}
                className="p-4 m-4 rounded flex-row items-center bg-[#1E1E1E] border border-gray-600"
            >
                <MaterialIcons name="archive" size={24} color={Colors.PRIMARY_TEXT} />
                <CustomText variant="headingMedium" className="ml-2 text-white">
                    Archived Tasks
                </CustomText>
            </TouchableOpacity>

            {/* Multi-select Options Bar (in flow, below header) */}
            {multiSelect && (
                <View className="flex-row justify-around p-4 bg-transparent">
                    <CustomButton
                        title="Select All"
                        onPress={() =>
                            setSelectedTasks(tasks.filter(task => !task.isArchived).map(task => task.id))
                        }
                        className="py-2 px-4 rounded bg-blue-500"
                        textStyle={{ color: Colors.PRIMARY_TEXT }}
                    />
                    <CustomButton
                        title="Deselect All"
                        onPress={() => setSelectedTasks([])}
                        className="py-2 px-4 rounded bg-blue-500"
                        textStyle={{ color: Colors.PRIMARY_TEXT }}
                    />
                    <CustomButton
                        title="More"
                        onPress={() => setMoreModalVisible(true)}
                        className="py-2 px-4 rounded bg-blue-500"
                        textStyle={{ color: Colors.PRIMARY_TEXT }}
                        icon={<MaterialIcons name="more-vert" size={20} color={Colors.PRIMARY_TEXT} />}
                    />
                </View>
            )}

            <FlatList
                data={tasks.filter(task => !task.isArchived)}
                keyExtractor={(item: Task) => item.id}
                renderItem={({ item }) => (
                    <TaskItem
                        task={item}
                        isSelected={selectedTasks.includes(item.id)}
                        onToggleSelect={toggleSelectTask}
                        multiSelect={multiSelect}
                        onActionComplete={refreshTasks}
                    />
                )}
                contentContainerStyle={{ paddingTop: multiSelect ? 16 : 16, padding: 16, paddingBottom: 120 }}
                showsVerticalScrollIndicator={false}
            />

            {/* Footer: Create Task Button */}
            <View className="absolute bottom-4 w-full border-t border-gray-600 py-8 px-6 bg-[#121212]">
                <CustomButton
                    title="Create Task"
                    onPress={() => router.push('/task/CreateTaskScreen')}
                    className="py-3 px-6 rounded bg-orange-500"
                    textStyle={{ color: Colors.PRIMARY_TEXT, textAlign: 'center', fontWeight: 'bold', fontSize: 20 }}
                />
            </View>

            {/* Modal for More Options in Multi-Select */}
            <Modal visible={moreModalVisible} transparent animationType="fade" onRequestClose={() => setMoreModalVisible(false)}>
                <TouchableOpacity className="flex-1 justify-center items-center bg-black/50" onPress={() => setMoreModalVisible(false)}>
                    <View className="bg-black p-4 rounded w-4/5">
                        <TouchableOpacity onPress={completeSelected} className="py-2 px-4 rounded bg-indigo-500 mt-2 flex-row items-center">
                            <View style={{ width: 30, alignItems: 'center' }}>
                                <MaterialIcons name="done-all" size={18} color={Colors.PRIMARY_TEXT} />
                            </View>
                            <CustomText variant="headingMedium" className="text-white ml-2">
                                Complete Selected
                            </CustomText>
                        </TouchableOpacity>
                        <TouchableOpacity onPress={deleteSelected} className="py-2 px-4 rounded bg-indigo-500 mt-2 flex-row items-center">
                            <View style={{ width: 30, alignItems: 'center' }}>
                                <MaterialIcons name="delete" size={18} color={Colors.PRIMARY_TEXT} />
                            </View>
                            <CustomText variant="headingMedium" className="text-white ml-2">
                                Delete Selected
                            </CustomText>
                        </TouchableOpacity>
                        <TouchableOpacity onPress={archiveSelected} className="py-2 px-4 rounded bg-indigo-500 mt-2 flex-row items-center">
                            <View style={{ width: 30, alignItems: 'center' }}>
                                <MaterialIcons name="archive" size={18} color={Colors.PRIMARY_TEXT} />
                            </View>
                            <CustomText variant="headingMedium" className="text-white ml-2">
                                Archive Selected
                            </CustomText>
                        </TouchableOpacity>
                    </View>
                </TouchableOpacity>
            </Modal>
        </SafeAreaView>
    );
}
