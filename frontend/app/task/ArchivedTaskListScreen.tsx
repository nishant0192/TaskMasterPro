import React from 'react';
import { SafeAreaView, FlatList, TouchableOpacity } from 'react-native';
import CustomText from '@/components/CustomText';
import { useTasksStore } from '@/store/tasksStore';
import { useRouter } from 'expo-router';
import { Ionicons, MaterialIcons, AntDesign } from '@expo/vector-icons';
import { Colors } from '@/constants/Colors';
import { Task } from '@/types/tasks';
import { useUpdateTask } from '@/api/task/useTask';
import { showAlert } from '@/components/CustomAlert';

const ArchivedTaskItem: React.FC<{ task: Task }> = ({ task }) => {
  const router = useRouter();
  const updateTaskMutation = useUpdateTask();
  // Retrieve the tasks store updater to update our local state optimistically.
  const { tasks, setTasks } = useTasksStore();

  const handleUnarchive = () => {
    updateTaskMutation.mutate(
      { id: task.id, isArchived: false },
      {
        onSuccess: () => {
          // Optimistically remove the task from the archived list.
          setTasks(tasks.filter((t: Task) => t.id !== task.id));
          showAlert({
            message: 'Task unarchived successfully.',
            type: 'success',
            title: 'Unarchived',
          });
        },
      }
    );
  };

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
      {/* Unarchive button with efficient API call */}
      <TouchableOpacity onPress={handleUnarchive} className="ml-2">
        <MaterialIcons name="unarchive" size={24} color={Colors.PRIMARY_TEXT} />
      </TouchableOpacity>
    </TouchableOpacity>
  );
};

export default function ArchivedTaskListScreen() {
  const { tasks } = useTasksStore();
  // Filter tasks that are archived.
  const archivedTasks = tasks.filter((task: Task) => task.isArchived);
  const router = useRouter();

  return (
    <SafeAreaView className="flex-1 bg-[#121212] relative">
      <TouchableOpacity
        onPress={() => router.replace('/task/TaskListScreen')}
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
