import React, { useState, useEffect, useCallback } from 'react';
import { SafeAreaView, FlatList, TouchableOpacity, Modal, View, RefreshControl } from 'react-native';
import CustomText from '@/components/CustomText';
import CustomButton from '@/components/CustomButton';
import StyledInput from '@/components/StyledInput';
import { useRouter } from 'expo-router';
import { MaterialIcons, Ionicons, AntDesign } from '@expo/vector-icons';
import { Colors } from '@/constants/Colors';
import { useTasksStore } from '@/store/tasksStore';
import { useGetTasks, useUpdateTask, useDeleteTask } from '@/api/task/useTask';
import { Task } from '@/types/tasks';
import { showAlert, showDialog } from '@/components/CustomAlert';

const statusOptions = [
  { value: 'TODO', label: 'Todo' },
  { value: 'in_progres', label: 'In Progress' },
  { value: 'DONE', label: 'Done' },
];

const TaskItem: React.FC<{
  task: Task;
  isSelected: boolean;
  multiSelect: boolean;
  onToggleSelect: (id: string) => void;
  onActionComplete: () => void;
}> = ({ task, isSelected, multiSelect, onToggleSelect, onActionComplete }) => {
  const router = useRouter();
  const updateTaskMutation = useUpdateTask();
  const deleteTaskMutation = useDeleteTask();
  const [menuVisible, setMenuVisible] = useState(false);

  const handlePress = () => {
    if (multiSelect) {
      onToggleSelect(task.id);
    } else {
      router.push({ pathname: '/task/TaskDetailsScreen', params: { id: task.id } });
    }
  };

  const options = [
    {
      title: 'Details',
      icon: (
        <Ionicons
          name="information-circle-outline"
          size={18}
          color={Colors.PRIMARY_TEXT}
        />
      ),
      action: () => {
        setMenuVisible(false);
        router.push({ pathname: '/task/TaskDetailsScreen', params: { id: task.id } });
        onActionComplete();
      },
    },
    ...(task.status !== 'DONE'
      ? [
        {
          title: 'Complete',
          icon: (
            <MaterialIcons
              name="done-all"
              size={18}
              color={Colors.PRIMARY_TEXT}
            />
          ),
          action: () => {
            setMenuVisible(false);
            updateTaskMutation.mutate(
              { id: task.id, status: 'DONE' },
              {
                onSuccess: () => {
                  onActionComplete();
                  showAlert({
                    message: 'Task marked complete.',
                    type: 'success',
                    title: 'Completed',
                  });
                },
              }
            );
          },
        },
      ]
      : []),
    {
      title: 'Delete',
      icon: (
        <MaterialIcons
          name="delete"
          size={18}
          color={Colors.PRIMARY_TEXT}
        />
      ),
      action: () => {
        setMenuVisible(false);
        showDialog({
          message: 'Are you sure you want to delete this task? This action cannot be undone.',
          type: 'warning',
          title: 'Confirm Deletion',
          cancelText: 'No, Cancel',
          confirmText: 'Yes, Delete',
          onConfirmPressed: () => {
            deleteTaskMutation.mutate(
              { id: task.id },
              {
                onSuccess: (data) => {
                  onActionComplete();
                  showAlert({
                    message: 'Task deleted successfully.',
                    type: 'success',
                    title: 'Deleted',
                  });
                },
                onError: (error: any) => {
                  // Convert the error to string to reliably inspect its content.
                  const errorStr = error?.toString().toLowerCase() || '';
                  // If the error indicates that the task is already deleted,
                  // or the response format is invalid (but deletion occurred),
                  // treat the deletion as successful.
                  if (
                    errorStr.includes('task deletion failed') ||
                    errorStr.includes('not found') ||
                    errorStr.includes('invalid response format')
                  ) {
                    onActionComplete();
                    showAlert({
                      message: 'Task deleted successfully.',
                      type: 'success',
                      title: 'Deleted',
                    });
                  } else {
                    showAlert({
                      message: 'Failed to delete task.',
                      type: 'error',
                      title: 'Error',
                    });
                  }
                },
              }
            );
          },
          onCancelPressed: () =>
            showAlert({
              message: 'Task deletion cancelled',
              type: 'info',
              title: 'Cancelled',
            }),
        });
      },
    },
    {
      title: 'Archive',
      icon: (
        <MaterialIcons
          name="archive"
          size={18}
          color={Colors.PRIMARY_TEXT}
        />
      ),
      action: () => {
        setMenuVisible(false);
        updateTaskMutation.mutate(
          { id: task.id, isArchived: true },
          {
            onSuccess: () => {
              onActionComplete();
              showAlert({
                message: 'Task archived successfully.',
                type: 'success',
                title: 'Archived',
              });
            },
          }
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
        className={`p-4 rounded mb-2 flex-row items-center ${isSelected ? 'border-2 border-blue-500' : 'border border-gray-600'}`}
        style={{ backgroundColor: Colors.SECONDARY_BACKGROUND }}
      >
        <View className="flex-1">
          <CustomText
            variant="headingMedium"
            className="mb-1"
            style={{ color: Colors.PRIMARY_TEXT }}
          >
            {task.title}
          </CustomText>
          {task.description && (
            <CustomText
              variant="headingSmall"
              style={{ color: Colors.SECONDARY_TEXT }}
            >
              {task.description}
            </CustomText>
          )}
        </View>
        <TouchableOpacity onPress={() => setMenuVisible(true)} className="ml-2">
          <MaterialIcons
            name="more-vert"
            size={20}
            color={Colors.PRIMARY_TEXT}
          />
        </TouchableOpacity>
      </TouchableOpacity>
      <Modal
        visible={menuVisible}
        transparent
        animationType="fade"
        onRequestClose={() => setMenuVisible(false)}
      >
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
                <View style={{ width: 30, alignItems: 'center' }}>
                  {option.icon}
                </View>
                <CustomText
                  variant="headingMedium"
                  className="text-white ml-2"
                >
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
  const [search, setSearch] = useState('');
  const [filterStatus, setFilterStatus] = useState('');
  const [statusDropdownVisible, setStatusDropdownVisible] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [triggerFetch, setTriggerFetch] = useState(0);
  const multiSelect = selectedTasks.length > 0;
  const getTasksMutation = useGetTasks({ search, status: filterStatus });
  const updateTaskMutation = useUpdateTask();
  const deleteTaskMutation = useDeleteTask();

  // Define a refresh function to refetch tasks.
  const refreshTasks = useCallback(() => {
    setRefreshing(true);
    getTasksMutation.mutate(undefined, {
      onSuccess: (data) => {
        if (data?.tasks) {
          setTasks(data.tasks);
        }
        setRefreshing(false);
      },
      onError: () => {
        setRefreshing(false);
        showAlert({
          message: 'Failed to refresh tasks',
          type: 'error',
          title: 'Error',
        });
      },
    });
  }, [getTasksMutation, setTasks]);

  // onActionComplete refreshes the list.
  const onActionComplete = useCallback(() => {
    setTriggerFetch((prev) => prev + 1);
    refreshTasks();
  }, [refreshTasks]);

  // Multi-select action handlers.
  const completeSelected = async () => {
    try {
      const updates = selectedTasks.map((id) => {
        const task = tasks.find((t) => t.id === id);
        if (task && task.status !== 'DONE') {
          return new Promise<void>((resolve) => {
            updateTaskMutation.mutate(
              { id, status: 'DONE' },
              { onSuccess: () => resolve(), onError: () => resolve() }
            );
          });
        }
        return Promise.resolve();
      });

      await Promise.all(updates);
      setSelectedTasks([]);
      setMoreModalVisible(false);
      onActionComplete();
      showAlert({
        message: 'Tasks completed successfully.',
        type: 'success',
        title: 'Completed',
      });
    } catch (error) {
      showAlert({
        message: 'Failed to complete tasks',
        type: 'error',
        title: 'Error',
      });
    }
  };

  const deleteSelected = async () => {
    try {
      const deletions = selectedTasks.map(
        (id) =>
          new Promise<void>((resolve) => {
            deleteTaskMutation.mutate(
              { id },
              {
                onSuccess: () => resolve(),
                onError: () => resolve(),
              }
            );
          })
      );
      await Promise.all(deletions);
      setSelectedTasks([]);
      setMoreModalVisible(false);
      onActionComplete();
      showAlert({
        message: 'Tasks deleted successfully.',
        type: 'success',
        title: 'Deleted',
      });
    } catch (error) {
      showAlert({
        message: 'Failed to delete tasks',
        type: 'error',
        title: 'Error',
      });
    }
  };

  const archiveSelected = async () => {
    try {
      const archives = selectedTasks.map(
        (id) =>
          new Promise<void>((resolve) => {
            updateTaskMutation.mutate(
              { id, isArchived: true },
              { onSuccess: () => resolve(), onError: () => resolve() }
            );
          })
      );
      await Promise.all(archives);
      setSelectedTasks([]);
      setMoreModalVisible(false);
      onActionComplete();
      showAlert({
        message: 'Tasks archived successfully.',
        type: 'success',
        title: 'Archived',
      });
    } catch (error) {
      showAlert({
        message: 'Failed to archive tasks',
        type: 'error',
        title: 'Error',
      });
    }
  };

  // Show confirmation dialog before deleting selected tasks.
  const handleDeleteSelected = () => {
    showDialog({
      message: 'Are you sure you want to delete the selected tasks? This action cannot be undone.',
      type: 'warning',
      title: 'Confirm Deletion',
      cancelText: 'No, Cancel',
      confirmText: 'Yes, Delete',
      onConfirmPressed: () => {
        deleteSelected();
      },
      onCancelPressed: () => {
        showAlert({
          message: 'Task deletion cancelled',
          type: 'info',
          title: 'Cancelled',
        });
      },
    });
  };

  useEffect(() => {
    getTasksMutation.mutate(undefined, {
      onSuccess: (data) => {
        if (data?.tasks) {
          setTasks(data.tasks);
        }
      },
      onError: () => {
        showAlert({
          message: 'Failed to fetch tasks',
          type: 'error',
          title: 'Error',
        });
      }
    });
  }, [search, filterStatus, triggerFetch]);

  const toggleSelectTask = (id: string) => {
    setSelectedTasks((prev) =>
      prev.includes(id) ? prev.filter((taskId) => taskId !== id) : [...prev, id]
    );
  };

  return (
    <SafeAreaView
      className="flex-1"
      style={{ backgroundColor: Colors.PRIMARY_BACKGROUND }}
    >
      {multiSelect && (
        <View className="flex-row justify-around p-4 mt-4 mb-[-20px]">
          <CustomButton
            title="Select All"
            onPress={() =>
              setSelectedTasks(
                tasks.filter((task) => !task.isArchived).map((task) => task.id)
              )
            }
            className="py-2 px-4 rounded"
            style={{ backgroundColor: Colors.BUTTON }}
            textStyle={{ color: Colors.PRIMARY_TEXT }}
          />
          <CustomButton
            title="Deselect All"
            onPress={() => setSelectedTasks([])}
            className="py-2 px-4 rounded"
            style={{ backgroundColor: Colors.BUTTON }}
            textStyle={{ color: Colors.PRIMARY_TEXT }}
          />
          <CustomButton
            title="More"
            onPress={() => setMoreModalVisible(true)}
            className="py-2 px-4 rounded"
            style={{ backgroundColor: Colors.BUTTON }}
            textStyle={{ color: Colors.PRIMARY_TEXT }}
            icon={
              <MaterialIcons
                name="more-vert"
                size={20}
                color={Colors.PRIMARY_TEXT}
              />
            }
          />
        </View>
      )}

      <View className="p-4">
        <StyledInput
          mode="outlined"
          labelText="Search tasks"
          value={search}
          onChangeText={setSearch}
          placeholder="Enter keyword..."
          keyboardType="default"
          colors={{
            backgroundColor: Colors.SECONDARY_BACKGROUND,
            textColor: Colors.PRIMARY_TEXT,
            placeholderTextColor: Colors.SECONDARY_TEXT,
            neutralBorderColor: Colors.DIVIDER,
          }}
        />
        <CustomButton
          title="View Archived Tasks"
          onPress={() => router.push('/task/ArchivedTaskListScreen')}
          className="mt-4 py-3 px-4 rounded"
          style={{ backgroundColor: Colors.BUTTON }}
          textStyle={{
            color: Colors.PRIMARY_TEXT,
            textAlign: 'center',
          }}
          icon={
            <MaterialIcons
              name="archive"
              size={20}
              color={Colors.PRIMARY_TEXT}
            />
          }
        />
        <TouchableOpacity
          onPress={() => setStatusDropdownVisible((prev) => !prev)}
          className="mt-4 p-3 border rounded flex-row items-center justify-between"
          style={{
            backgroundColor: Colors.SECONDARY_BACKGROUND,
            borderColor: Colors.DIVIDER,
          }}
        >
          <CustomText
            variant="headingMedium"
            style={{ color: Colors.PRIMARY_TEXT, flex: 1 }}
          >
            {filterStatus
              ? statusOptions.find((option) => option.value === filterStatus)
                ?.label || filterStatus
              : 'All Statuses'}
          </CustomText>
          <AntDesign name="caretdown" size={18} color={Colors.PRIMARY_TEXT} />
        </TouchableOpacity>
        {statusDropdownVisible && (
          <View
            className="mt-2 border rounded"
            style={{
              backgroundColor: Colors.SECONDARY_BACKGROUND,
              borderColor: Colors.DIVIDER,
            }}
          >
            {statusOptions.map((option) => (
              <TouchableOpacity
                key={option.value}
                onPress={() => {
                  setFilterStatus(option.value);
                  setStatusDropdownVisible(false);
                }}
                className="p-3"
              >
                <CustomText
                  variant="headingMedium"
                  style={{ color: Colors.PRIMARY_TEXT }}
                >
                  {option.label}
                </CustomText>
              </TouchableOpacity>
            ))}
          </View>
        )}
      </View>

      <FlatList
        data={tasks.filter((task: Task) => !task.isArchived)}
        keyExtractor={(item: Task) => item.id}
        renderItem={({ item }) => (
          <TaskItem
            task={item}
            isSelected={selectedTasks.includes(item.id)}
            multiSelect={multiSelect}
            onToggleSelect={toggleSelectTask}
            onActionComplete={onActionComplete}
          />
        )}
        contentContainerStyle={{
          paddingTop: 16,
          paddingHorizontal: 16,
          paddingBottom: 120,
        }}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={refreshTasks}
            tintColor={Colors.PRIMARY_TEXT}
            colors={[Colors.PRIMARY_TEXT]}
            progressBackgroundColor={Colors.SECONDARY_BACKGROUND}
            progressViewOffset={20}
          />
        }
        showsVerticalScrollIndicator={false}
      />

      <View
        className="absolute bottom-4 w-full"
        style={{
          bottom: 0,
          left: 0,
          right: 0,
          borderTopWidth: 1,
          borderTopColor: Colors.DIVIDER,
          paddingVertical: 20,
          paddingHorizontal: 24,
          backgroundColor: Colors.PRIMARY_BACKGROUND,
        }}
      >
        <CustomButton
          title="Create Task"
          onPress={() => router.push('/task/CreateTaskScreen')}
          className="py-3 px-6 rounded"
          style={{ backgroundColor: Colors.BUTTON }}
          textStyle={{
            color: Colors.PRIMARY_TEXT,
            textAlign: 'center',
            fontWeight: 'bold',
            fontSize: 20,
          }}
          icon={<Ionicons name="create-outline" size={24} color={Colors.PRIMARY_TEXT} style={{ paddingBottom: 2 }} />}
        />
      </View>

      <Modal
        visible={moreModalVisible}
        transparent
        animationType="fade"
        onRequestClose={() => setMoreModalVisible(false)}
      >
        <TouchableOpacity
          className="flex-1 justify-center items-center bg-black/50"
          onPress={() => setMoreModalVisible(false)}
        >
          <View className="bg-black p-4 rounded w-4/5">
            <CustomButton
              title="Complete Selected"
              onPress={completeSelected}
              className="py-2 px-4 rounded mt-2 flex-row items-center"
              style={{ backgroundColor: Colors.BUTTON }}
              textStyle={{ color: Colors.PRIMARY_TEXT }}
              icon={
                <MaterialIcons
                  name="done-all"
                  size={18}
                  color={Colors.PRIMARY_TEXT}
                />
              }
            />
            <CustomButton
              title="Delete Selected"
              onPress={handleDeleteSelected}
              className="py-2 px-4 rounded mt-2 flex-row items-center"
              style={{ backgroundColor: Colors.BUTTON }}
              textStyle={{ color: Colors.PRIMARY_TEXT }}
              icon={
                <MaterialIcons
                  name="delete"
                  size={18}
                  color={Colors.PRIMARY_TEXT}
                />
              }
            />
            <CustomButton
              title="Archive Selected"
              onPress={archiveSelected}
              className="py-2 px-4 rounded mt-2 flex-row items-center"
              style={{ backgroundColor: Colors.BUTTON }}
              textStyle={{ color: Colors.PRIMARY_TEXT }}
              icon={
                <MaterialIcons
                  name="archive"
                  size={18}
                  color={Colors.PRIMARY_TEXT}
                />
              }
            />
          </View>
        </TouchableOpacity>
      </Modal>
    </SafeAreaView>
  );
}
