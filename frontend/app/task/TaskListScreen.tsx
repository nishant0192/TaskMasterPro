// TaskListScreen.tsx
import React, { useState, useEffect } from 'react';
import { SafeAreaView, FlatList, TouchableOpacity, Modal, View } from 'react-native';
import CustomText from '@/components/CustomText';
import CustomButton from '@/components/CustomButton';
import StyledInput from '@/components/StyledInput';
import { useRouter } from 'expo-router';
import { MaterialIcons, Ionicons, AntDesign } from '@expo/vector-icons';
import { Colors } from '@/constants/Colors';
import { useTasksStore } from '@/store/tasksStore';
import { useGetTasks, useUpdateTask, useDeleteTask } from '@/api/task/useTask';
import { Task } from '@/types/tasks';

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
      icon: <Ionicons name="information-circle-outline" size={18} color={Colors.PRIMARY_TEXT} />,
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
          icon: <MaterialIcons name="done-all" size={18} color={Colors.PRIMARY_TEXT} />,
          action: () => {
            setMenuVisible(false);
            updateTaskMutation.mutate({ id: task.id, status: 'DONE' }, { onSuccess: onActionComplete });
          },
        },
      ]
      : []),
    {
      title: 'Delete',
      icon: <MaterialIcons name="delete" size={18} color={Colors.PRIMARY_TEXT} />,
      action: () => {
        setMenuVisible(false);
        deleteTaskMutation.mutate({ id: task.id }, { onSuccess: onActionComplete });
      },
    },
    {
      title: 'Archive',
      icon: <MaterialIcons name="archive" size={18} color={Colors.PRIMARY_TEXT} />,
      action: () => {
        setMenuVisible(false);
        updateTaskMutation.mutate({ id: task.id, isArchived: true }, { onSuccess: onActionComplete });
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
          <CustomText variant="headingMedium" className="mb-1" style={{ color: Colors.PRIMARY_TEXT }}>
            {task.title}
          </CustomText>
          {task.description && (
            <CustomText variant="headingSmall" style={{ color: Colors.SECONDARY_TEXT }}>
              {task.description}
            </CustomText>
          )}
        </View>
        <TouchableOpacity onPress={() => setMenuVisible(true)} className="ml-2">
          <MaterialIcons name="more-vert" size={20} color={Colors.PRIMARY_TEXT} />
        </TouchableOpacity>
      </TouchableOpacity>
      <Modal visible={menuVisible} transparent animationType="fade" onRequestClose={() => setMenuVisible(false)}>
        <TouchableOpacity className="flex-1 justify-center items-center bg-black/50" onPress={() => setMenuVisible(false)}>
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
  const [search, setSearch] = useState('');
  const [filterStatus, setFilterStatus] = useState('');
  const [statusDropdownVisible, setStatusDropdownVisible] = useState(false);
  const [triggerFetch, setTriggerFetch] = useState(0);
  const multiSelect = selectedTasks.length > 0;
  const getTasksMutation = useGetTasks({ search, status: filterStatus });
  const updateTaskMutation = useUpdateTask();
  const deleteTaskMutation = useDeleteTask();

  // Multi-select action handlers.
  const completeSelected = async () => {
    selectedTasks.forEach(id => {
      const task = tasks.find(t => t.id === id);
      if (task && task.status !== 'DONE') {
        updateTaskMutation.mutate({ id, status: 'DONE' });
      }
    });
    setSelectedTasks([]);
    setMoreModalVisible(false);
    setTriggerFetch(prev => prev + 1);
  };

  const deleteSelected = async () => {
    selectedTasks.forEach(id => {
      deleteTaskMutation.mutate({ id });
    });
    setSelectedTasks([]);
    setMoreModalVisible(false);
    setTriggerFetch(prev => prev + 1);
  };

  const archiveSelected = async () => {
    selectedTasks.forEach(id => {
      updateTaskMutation.mutate({ id, isArchived: true });
    });
    setSelectedTasks([]);
    setMoreModalVisible(false);
    setTriggerFetch(prev => prev + 1);
  };

  useEffect(() => {
    getTasksMutation.mutate(undefined, {
      onSuccess: (data) => {
        if (data?.tasks) {
          setTasks(data.tasks);
        }
      },
    });
  }, [search, filterStatus, triggerFetch]);

  const toggleSelectTask = (id: string) => {
    setSelectedTasks(prev =>
      prev.includes(id) ? prev.filter(taskId => taskId !== id) : [...prev, id]
    );
  };

  return (
    <SafeAreaView className="flex-1" style={{ backgroundColor: Colors.PRIMARY_BACKGROUND }}>
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
        {/* Button to view archived tasks using CustomButton */}
        <CustomButton
          title="View Archived Tasks"
          onPress={() => router.push('/task/ArchivedTaskListScreen')}
          className="mt-4 py-3 px-4 rounded"
          style={{ backgroundColor: Colors.BUTTON }}
          textStyle={{ color: Colors.PRIMARY_TEXT, textAlign: 'center' }}
          icon={<MaterialIcons name="archive" size={20} color={Colors.PRIMARY_TEXT} />}
        />
        <TouchableOpacity
          onPress={() => setStatusDropdownVisible(prev => !prev)}
          className="mt-4 p-3 border rounded flex-row items-center justify-between"
          style={{ backgroundColor: Colors.SECONDARY_BACKGROUND, borderColor: Colors.DIVIDER }}
        >
          <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT, flex: 1 }}>
            {filterStatus ? (statusOptions.find(option => option.value === filterStatus)?.label || filterStatus) : 'All Statuses'}
          </CustomText>
          <AntDesign name="caretdown" size={18} color={Colors.PRIMARY_TEXT} />
        </TouchableOpacity>
        {statusDropdownVisible && (
          <View
            className="mt-2 border rounded"
            style={{ backgroundColor: Colors.SECONDARY_BACKGROUND, borderColor: Colors.DIVIDER }}
          >
            {statusOptions.map(option => (
              <TouchableOpacity
                key={option.value}
                onPress={() => {
                  setFilterStatus(option.value);
                  setStatusDropdownVisible(false);
                }}
                className="p-3"
              >
                <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT }}>
                  {option.label}
                </CustomText>
              </TouchableOpacity>
            ))}
          </View>
        )}
      </View>

      {multiSelect && (
        <View className="flex-row justify-around p-4 mt-4 mb-[-20px]">
          <CustomButton
            title="Select All"
            onPress={() =>
              setSelectedTasks(tasks.filter(task => !task.isArchived).map(task => task.id))
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
            icon={<MaterialIcons name="more-vert" size={20} color={Colors.PRIMARY_TEXT} />}
          />
        </View>
      )}

      <FlatList
        data={tasks.filter((task: Task) => !task.isArchived)}
        keyExtractor={(item: Task) => item.id}
        renderItem={({ item }) => (
          <TaskItem
            task={item}
            isSelected={selectedTasks.includes(item.id)}
            multiSelect={multiSelect}
            onToggleSelect={toggleSelectTask}
            onActionComplete={() => setTriggerFetch(prev => prev + 1)}
          />
        )}
        contentContainerStyle={{ paddingTop: 16, paddingHorizontal: 16, paddingBottom: 120 }}
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
          textStyle={{ color: Colors.PRIMARY_TEXT, textAlign: 'center', fontWeight: 'bold', fontSize: 20 }}
        />
      </View>

      <Modal visible={moreModalVisible} transparent animationType="fade" onRequestClose={() => setMoreModalVisible(false)}>
        <TouchableOpacity className="flex-1 justify-center items-center bg-black/50" onPress={() => setMoreModalVisible(false)}>
          <View className="bg-black p-4 rounded w-4/5">
            <CustomButton
              title="Complete Selected"
              onPress={completeSelected}
              className="py-2 px-4 rounded mt-2 flex-row items-center"
              style={{ backgroundColor: Colors.BUTTON }}
              textStyle={{ color: Colors.PRIMARY_TEXT }}
              icon={<MaterialIcons name="done-all" size={18} color={Colors.PRIMARY_TEXT} />}
            />
            <CustomButton
              title="Delete Selected"
              onPress={deleteSelected}
              className="py-2 px-4 rounded mt-2 flex-row items-center"
              style={{ backgroundColor: Colors.BUTTON }}
              textStyle={{ color: Colors.PRIMARY_TEXT }}
              icon={<MaterialIcons name="delete" size={18} color={Colors.PRIMARY_TEXT} />}
            />
            <CustomButton
              title="Archive Selected"
              onPress={archiveSelected}
              className="py-2 px-4 rounded mt-2 flex-row items-center"
              style={{ backgroundColor: Colors.BUTTON }}
              textStyle={{ color: Colors.PRIMARY_TEXT }}
              icon={<MaterialIcons name="archive" size={18} color={Colors.PRIMARY_TEXT} />}
            />
          </View>
        </TouchableOpacity>
      </Modal>
    </SafeAreaView>
  );
}
