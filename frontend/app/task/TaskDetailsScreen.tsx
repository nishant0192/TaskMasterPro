import React, { useState, useEffect } from 'react';
import {
  SafeAreaView,
  ScrollView,
  View,
  Alert,
  TouchableOpacity,
  Switch,
} from 'react-native';
import { useRouter, useLocalSearchParams } from 'expo-router';
import CustomText from '@/components/CustomText';
import StyledInput from '@/components/StyledInput';
import CustomButton from '@/components/CustomButton';
import CustomDatePicker from '@/components/CustomDatePicker';
import { useGetTaskById, useUpdateTask } from '@/api/task/useTask';
import { Colors } from '@/constants/Colors';
import { AntDesign } from '@expo/vector-icons';
import { useSharedValue } from 'react-native-reanimated';
import { Slider as AwesomeSlider } from 'react-native-awesome-slider';
import * as Haptics from 'expo-haptics';

const statusOptions = [
  { value: 'TODO', label: 'Todo' },
  { value: 'in_progres', label: 'In progress' },
  { value: 'DONE', label: 'Done' },
];

const motivatingSentences: { [key: number]: string } = {
  0: "Let's get started!",
  10: "Keep it up!",
  20: "You're making progress!",
  30: "Great effort!",
  40: "Almost halfway there!",
  50: "Halfway done, keep pushing!",
  60: "You're doing fantastic!",
  70: "Keep going strong!",
  80: "Almost at the finish line!",
  90: "Just a little more effort!",
  100: "Amazing! You've reached your goal!",
};

export default function TaskDetailsScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const { data, isLoading, error } = useGetTaskById(id);
  const updateTaskMutation = useUpdateTask();
  const router = useRouter();

  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [dueDate, setDueDate] = useState<Date | undefined>(undefined);
  const [priority, setPriority] = useState('');
  const [status, setStatus] = useState('');
  const [progress, setProgress] = useState('0');
  const [isArchived, setIsArchived] = useState<boolean>(false);
  const [datePickerVisible, setDatePickerVisible] = useState(false);
  const [statusDropdownVisible, setStatusDropdownVisible] = useState(false);

  const progressShared = useSharedValue(Number(progress));
  const minValue = useSharedValue(0);
  const maxValue = useSharedValue(100);

  useEffect(() => {
    if (data?.task) {
      setTitle(data.task.title);
      setDescription(data.task.description || '');
      setDueDate(data.task.dueDate ? new Date(data.task.dueDate) : undefined);
      setPriority(data.task.priority ? String(data.task.priority) : '');
      setStatus(data.task.status === 'IN_PROGRESS' ? 'in_progres' : data.task.status || '');
      setProgress(data.task.progress ? String(data.task.progress) : '0');
      setIsArchived(data.task.isArchived);
      progressShared.value = data.task.progress ? Number(data.task.progress) : 0;
    }
  }, [data]);

  // When status becomes "DONE", ensure the slider shows 100%
  useEffect(() => {
    if (status === 'DONE' && Number(progress) !== 100) {
      setProgress("100");
      progressShared.value = 100;
    }
  }, [status]);

  const handleUpdateTask = () => {
    if (!title) {
      Alert.alert('Error', 'Title is required');
      return;
    }

    const updateData: any = {
      id,
      title,
      description,
      dueDate: dueDate ? dueDate.toISOString() : undefined,
      priority: priority ? Number(priority) : undefined,
      status,
      progress: progress ? Number(progress) : undefined,
      isArchived,
    };

    if (status === 'DONE') {
      updateData.completedAt = new Date().toISOString();
    }

    updateTaskMutation.mutate(updateData, {
      onSuccess: () => {
        console.log('Task updated');
        router.push('/task/TaskListScreen');
      },
      onError: (err) => {
        console.error('Error updating task', err);
        Alert.alert('Error', 'Failed to update task');
      },
    });
  };

  if (isLoading) {
    return (
      <SafeAreaView style={{
        flex: 1,
        backgroundColor: Colors.PRIMARY_BACKGROUND,
        padding: 24,
        justifyContent: 'center',
        alignItems: 'center'
      }}>
        <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT }}>
          Loading task details...
        </CustomText>
      </SafeAreaView>
    );
  }

  if (error) {
    return (
      <SafeAreaView style={{
        flex: 1,
        backgroundColor: Colors.PRIMARY_BACKGROUND,
        padding: 24,
        justifyContent: 'center',
        alignItems: 'center'
      }}>
        <CustomText variant="headingMedium" style={{ color: Colors.ERROR, textAlign: 'center' }}>
          Error fetching task details: {error.message}
        </CustomText>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: Colors.PRIMARY_BACKGROUND, padding: 24, position: 'relative' }}>
      <ScrollView contentContainerStyle={{ flexGrow: 1, paddingBottom: 160 }} showsVerticalScrollIndicator={false}>
        {/* Title */}
        <View style={{ marginBottom: 24 }}>
          <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT, marginBottom: 8 }}>
            Title:
          </CustomText>
          <StyledInput
            mode="outlined"
            labelText="Title"
            value={title}
            onChangeText={setTitle}
            placeholder="Enter task title"
            keyboardType="default"
            colors={{
              backgroundColor: Colors.SECONDARY_BACKGROUND,
              textColor: Colors.PRIMARY_TEXT,
              placeholderTextColor: Colors.SECONDARY_TEXT,
              neutralBorderColor: Colors.DIVIDER,
            }}
          />
        </View>

        {/* Description */}
        <View style={{ marginBottom: 24 }}>
          <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT, marginBottom: 8 }}>
            Description:
          </CustomText>
          <StyledInput
            mode="outlined"
            labelText="Description"
            value={description}
            onChangeText={setDescription}
            placeholder="Enter task description"
            keyboardType="default"
            multiline
            numberOfLines={4}
            colors={{
              backgroundColor: Colors.SECONDARY_BACKGROUND,
              textColor: Colors.PRIMARY_TEXT,
              placeholderTextColor: Colors.SECONDARY_TEXT,
              neutralBorderColor: Colors.DIVIDER,
            }}
          />
        </View>

        {/* Due Date */}
        <View style={{ marginBottom: 24 }}>
          <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT, marginBottom: 8 }}>
            Due Date:
          </CustomText>
          <TouchableOpacity
            onPress={() => setDatePickerVisible(true)}
            style={{
              marginVertical: 8,
              paddingVertical: 12,
              paddingHorizontal: 16,
              borderWidth: 1,
              borderColor: Colors.DIVIDER,
              borderRadius: 8,
              justifyContent: 'center',
              backgroundColor: Colors.SECONDARY_BACKGROUND,
              flexDirection: 'row',
              alignItems: 'center',
            }}
          >
            <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT, flex: 1 }}>
              {dueDate ? dueDate.toLocaleDateString() : 'Select Due Date'}
            </CustomText>
            <AntDesign name="calendar" size={24} color={Colors.PRIMARY_TEXT} />
          </TouchableOpacity>
        </View>

        {/* Priority */}
        <View style={{ marginBottom: 24 }}>
          <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT, marginBottom: 8 }}>
            Priority (1-5):
          </CustomText>
          <StyledInput
            mode="outlined"
            labelText="Priority"
            value={priority}
            onChangeText={(text) => {
              let numericText = text.replace(/[^0-9]/g, '');
              if (numericText) {
                const num = parseInt(numericText, 10);
                if (num < 1) numericText = '1';
                if (num > 5) numericText = '5';
              }
              setPriority(numericText);
            }}
            placeholder="Enter a number between 1 and 5"
            keyboardType="numeric"
            colors={{
              backgroundColor: Colors.SECONDARY_BACKGROUND,
              textColor: Colors.PRIMARY_TEXT,
              placeholderTextColor: Colors.SECONDARY_TEXT,
              neutralBorderColor: Colors.DIVIDER,
            }}
          />
        </View>

        {/* Status */}
        <View style={{ marginBottom: 24 }}>
          <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT, marginBottom: 8 }}>
            Status:
          </CustomText>
          <TouchableOpacity
            onPress={() => setStatusDropdownVisible((prev) => !prev)}
            style={{
              paddingVertical: 12,
              paddingHorizontal: 16,
              borderWidth: 1,
              borderColor: Colors.DIVIDER,
              borderRadius: 8,
              backgroundColor: Colors.SECONDARY_BACKGROUND,
              flexDirection: 'row',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}
          >
            <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT, flex: 1 }}>
              {status ? (statusOptions.find(option => option.value === status)?.label || status) : 'Select Status'}
            </CustomText>
            <AntDesign name="caretdown" size={18} color={Colors.PRIMARY_TEXT} />
          </TouchableOpacity>
          {statusDropdownVisible && (
            <View
              style={{
                marginTop: 8,
                borderWidth: 1,
                borderColor: Colors.DIVIDER,
                borderRadius: 8,
                backgroundColor: Colors.SECONDARY_BACKGROUND,
              }}
            >
              {statusOptions.map((option) => (
                <TouchableOpacity
                  key={option.value}
                  onPress={() => {
                    setStatus(option.value);
                    setStatusDropdownVisible(false);
                  }}
                  style={{ paddingVertical: 12, paddingHorizontal: 16 }}
                >
                  <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT }}>
                    {option.label}
                  </CustomText>
                </TouchableOpacity>
              ))}
            </View>
          )}
        </View>

        {/* Progress with Awesome Slider */}
        <View style={{ marginBottom: 24 }}>
          <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT, marginBottom: 8 }}>
            Progress (%):
          </CustomText>
          <AwesomeSlider
            style={{ height: 10 }}
            containerStyle={{ borderRadius: 8 }}
            progress={progressShared}
            minimumValue={minValue}
            maximumValue={maxValue}
            step={10}
            onValueChange={(value: number) => {
              const snappedValue = Math.round(value / 10) * 10;
              setProgress(String(snappedValue));
              progressShared.value = snappedValue;
              if (snappedValue === 100 && status !== 'DONE') {
                setStatus('DONE');
                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
              } else if (snappedValue >= 10 && snappedValue < 100) {
                setStatus('in_progres');
              } else if (snappedValue < 10) {
                setStatus('TODO');
              }
            }}
            theme={{
              minimumTrackTintColor: Colors.BUTTON,
              maximumTrackTintColor: Colors.DIVIDER,
              bubbleBackgroundColor: Colors.SECONDARY_BACKGROUND,
            }}
          />
          <CustomText style={{ color: Colors.PRIMARY_TEXT, textAlign: 'center', marginTop: 8 }} variant="headingBig">
            {progress ? progress : 0}%
          </CustomText>
          <CustomText style={{ color: Colors.PRIMARY_TEXT, textAlign: 'center', marginTop: 4 }} variant="headingBig">
            {motivatingSentences[Number(progress)] || ""}
          </CustomText>
        </View>

        {/* isArchived using a Switch */}
        <View style={{ marginBottom: 24, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' }}>
          <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT }}>
            Archived:
          </CustomText>
          <Switch
            value={isArchived}
            onValueChange={setIsArchived}
            thumbColor={Colors.BUTTON}
            trackColor={{ false: Colors.DIVIDER, true: Colors.BUTTON }}
          />
        </View>
      </ScrollView>
      <View style={{ position: 'absolute', bottom: 0, left: 0, right: 0, borderTopWidth: 1, borderTopColor: Colors.DIVIDER, paddingVertical: 20, paddingHorizontal: 24, backgroundColor: Colors.PRIMARY_BACKGROUND }}>
        <CustomButton
          title="Update Task"
          onPress={handleUpdateTask}
          style={{ backgroundColor: Colors.BUTTON, paddingVertical: 12, borderRadius: 8 }}
          textStyle={{ color: Colors.PRIMARY_TEXT, textAlign: 'center', fontWeight: 'bold', fontSize: 20 }}
        />
      </View>
      {datePickerVisible && (
        <CustomDatePicker
          date={dueDate || new Date()}
          mode="date"
          minimumDate={new Date()}
          onConfirm={(date: Date) => {
            setDueDate(date);
            setDatePickerVisible(false);
          }}
          onCancel={() => setDatePickerVisible(false)}
        />
      )}
    </SafeAreaView>
  );
}
