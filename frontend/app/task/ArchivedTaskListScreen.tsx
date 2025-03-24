import React from 'react';
import { SafeAreaView, FlatList, TouchableOpacity } from 'react-native';
import CustomText from '@/components/CustomText';
import { useTasksStore } from '@/store/tasksStore';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { Colors } from '@/constants/Colors';
import { Task } from '@/types/tasks';
import { useUpdateTask, useGetTasks } from '@/api/task/useTask';

const ArchivedTaskItem: React.FC<{ task: Task }> = ({ task }) => {
  const router = useRouter();
  const updateTaskMutation = useUpdateTask();
  const { refetch } = useGetTasks();

  return (
    <TouchableOpacity
      onPress={() =>
        router.push({
          pathname: '/task/TaskDetailsScreen',
          params: { id: task.id },
        })
      }
      className="p-4 rounded mb-2 flex-row items-center bg-[#1E1E1E] border border-gray-600"
    >
      <TouchableOpacity
        className="flex-1"
        onPress={() =>
          router.push({
            pathname: '/task/TaskDetailsScreen',
            params: { id: task.id },
          })
        }
      >
        <CustomText variant="headingMedium" className="mb-1 text-white">
          {task.title}
        </CustomText>
        {task.description && (
          <CustomText variant="headingSmall" className="text-gray-300">
            {task.description}
          </CustomText>
        )}
      </TouchableOpacity>
      {/* Unarchive button: calls API and refreshes */}
      <TouchableOpacity
        onPress={() =>
          updateTaskMutation.mutate(
            { id: task.id, isArchived: false },
            { onSuccess: () => refetch() }
          )
        }
        className="ml-2"
      >
        <Ionicons name="arrow-up-circle" size={24} color={Colors.PRIMARY_TEXT} />
      </TouchableOpacity>
    </TouchableOpacity>
  );
};

export default function ArchivedTaskListScreen() {
  const { tasks } = useTasksStore();
  const archivedTasks = tasks.filter(task => task.isArchived);
  const router = useRouter();

  return (
    <SafeAreaView className="flex-1 bg-[#121212] relative">
      <TouchableOpacity
        onPress={() => router.push('/task/TaskListScreen')}
        className="p-4 m-4 rounded flex-row items-center bg-[#1E1E1E] border border-gray-600"
      >
        <Ionicons name="arrow-back" size={24} color={Colors.PRIMARY_TEXT} />
        <CustomText variant="headingMedium" className="ml-2 text-white">
          Back to Active Tasks
        </CustomText>
      </TouchableOpacity>
      <FlatList
        data={archivedTasks}
        keyExtractor={(item: Task) => item.id}
        renderItem={({ item }) => <ArchivedTaskItem task={item} />}
        contentContainerStyle={{ padding: 16 }}
        showsVerticalScrollIndicator={false}
      />
    </SafeAreaView>
  );
}
