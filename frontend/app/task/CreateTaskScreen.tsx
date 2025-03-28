// CreateTaskScreen.tsx
import React, { useState } from 'react';
import { SafeAreaView, Alert, View, TouchableOpacity, ScrollView } from 'react-native';
import CustomText from '@/components/CustomText';
import CustomButton from '@/components/CustomButton';
import StyledInput from '@/components/StyledInput';
import CustomDatePicker from '@/components/CustomDatePicker';
import { useCreateTask, useCreateAttachment } from '@/api/task/useTask';
import { useRouter } from 'expo-router';
import { Colors } from '@/constants/Colors';
import { AntDesign, MaterialIcons } from '@expo/vector-icons';
import * as DocumentPicker from 'expo-document-picker';
import * as FileSystem from 'expo-file-system';

export default function CreateTaskScreen() {
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [priority, setPriority] = useState('');
  const [dueDate, setDueDate] = useState<Date | undefined>(undefined);
  const [reminderAt, setReminderAt] = useState<Date | undefined>(undefined);
  const [datePickerVisible, setDatePickerVisible] = useState(false);
  const [reminderPickerVisible, setReminderPickerVisible] = useState(false);
  const [newSubtask, setNewSubtask] = useState('');
  const [subtasks, setSubtasks] = useState<string[]>([]);
  const [attachments, setAttachments] = useState<{ name: string; uri: string }[]>([]);

  const router = useRouter();
  const { mutate: createTask, status } = useCreateTask();
  const { mutate: createAttachment } = useCreateAttachment();
  const isMutating = status === 'pending';

  // Function to add a new attachment via DocumentPicker.
  const handleAddNewAttachment = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({});
      if (result.canceled) return;
      if (result.assets && result.assets[0]) {
        setAttachments((prev) => [
          ...prev,
          { name: result.assets[0].name, uri: result.assets[0].uri },
        ]);
      }
    } catch (err) {
      console.error('Attachment error:', err);
      Alert.alert('Error', 'Failed to add attachment');
    }
  };

  const handleCreateTask = async () => {
    if (!title) {
      Alert.alert('Error', 'Title is required');
      return;
    }

    // Create the task without attachments
    const payload = {
      title,
      description,
      dueDate: dueDate ? dueDate.toISOString() : undefined,
      priority: priority ? Number(priority) : undefined,
      reminderAt: reminderAt ? reminderAt.toISOString() : undefined,
      subtasks: subtasks.length > 0 ? subtasks.map((st) => ({ title: st })) : undefined,
    };

    createTask(payload, {
      onSuccess: async (data) => {
        console.log('Task created successfully', data);
        // Extract taskId from response (adjust based on your response structure)
        const taskId = data.id || data.task?.id;
        if (taskId && attachments.length > 0) {
          await Promise.all(
            attachments.map(async (att) => {
              const fileData = await FileSystem.readAsStringAsync(att.uri, {
                encoding: FileSystem.EncodingType.Base64,
              });
              createAttachment({
                taskId,
                fileName: att.name,
                fileData: fileData,
              });
            })
          );
        }
        router.push('/task/TaskListScreen');
      },
      onError: (err) => {
        console.error('Error creating task', err);
        Alert.alert('Error', 'Failed to create task');
      },
    });
  };

  const addSubtask = () => {
    if (!newSubtask.trim()) {
      Alert.alert('Error', 'Subtask title cannot be empty');
      return;
    }
    setSubtasks((prev) => [...prev, newSubtask.trim()]);
    setNewSubtask('');
  };

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: Colors.PRIMARY_BACKGROUND, padding: 24 }}>
      <ScrollView contentContainerStyle={{ paddingBottom: 160 }} keyboardShouldPersistTaps="handled" showsVerticalScrollIndicator={false}>
        <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT, marginBottom: 16 }}>
          Create Task
        </CustomText>

        {/* Title Input */}
        <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT }}>
          Title:
        </CustomText>
        <StyledInput
          mode="outlined"
          value={title}
          labelText="Title"
          onChangeText={setTitle}
          placeholder="Task title"
          keyboardType="default"
          colors={{
            backgroundColor: Colors.SECONDARY_BACKGROUND,
            textColor: Colors.PRIMARY_TEXT,
            placeholderTextColor: Colors.SECONDARY_TEXT,
            neutralBorderColor: Colors.DIVIDER,
          }}
          style={{ marginBottom: 16 }}
        />

        {/* Description Input */}
        <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT, marginBottom: 8 }}>
          Description:
        </CustomText>
        <StyledInput
          mode="outlined"
          labelText="Description"
          value={description}
          onChangeText={setDescription}
          placeholder="Task description"
          keyboardType="default"
          multiline
          numberOfLines={10}
          colors={{
            backgroundColor: Colors.SECONDARY_BACKGROUND,
            textColor: Colors.PRIMARY_TEXT,
            placeholderTextColor: Colors.SECONDARY_TEXT,
            neutralBorderColor: Colors.DIVIDER,
          }}
          style={{ marginBottom: 16 }}
        />

        {/* Due Date Picker */}
        <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT, marginBottom: 8 }}>
          Due Date:
        </CustomText>
        <TouchableOpacity
          onPress={() => setDatePickerVisible(true)}
          style={{
            paddingVertical: 12,
            paddingHorizontal: 16,
            borderWidth: 1,
            borderColor: Colors.DIVIDER,
            borderRadius: 8,
            justifyContent: 'center',
            backgroundColor: Colors.SECONDARY_BACKGROUND,
            flexDirection: 'row',
            alignItems: 'center',
            marginBottom: 16,
          }}
        >
          <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT, flex: 1 }}>
            {dueDate ? dueDate.toLocaleDateString() : 'Select Due Date'}
          </CustomText>
          <AntDesign name="calendar" size={24} color={Colors.PRIMARY_TEXT} />
        </TouchableOpacity>

        {/* Reminder Picker */}
        <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT, marginBottom: 8 }}>
          Reminder:
        </CustomText>
        <TouchableOpacity
          onPress={() => setReminderPickerVisible(true)}
          style={{
            paddingVertical: 12,
            paddingHorizontal: 16,
            borderWidth: 1,
            borderColor: Colors.DIVIDER,
            borderRadius: 8,
            justifyContent: 'center',
            backgroundColor: Colors.SECONDARY_BACKGROUND,
            flexDirection: 'row',
            alignItems: 'center',
            marginBottom: 16,
          }}
        >
          <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT, flex: 1 }}>
            {reminderAt ? reminderAt.toLocaleDateString() : 'Select Reminder Date'}
          </CustomText>
          <AntDesign name="calendar" size={24} color={Colors.PRIMARY_TEXT} />
        </TouchableOpacity>

        {/* Priority Input */}
        <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT, marginBottom: 8 }}>
          Priority (1-5):
        </CustomText>
        <StyledInput
          mode="outlined"
          labelText="Priority (1-5)"
          value={priority}
          onChangeText={(text) => {
            let numericText = text.replace(/[^0-9]/g, '');
            const num = parseInt(numericText, 10);
            if (!isNaN(num)) {
              if (num >= 1 && num <= 5) {
                setPriority(numericText);
              } else {
                setPriority(num > 5 ? '5' : '1');
              }
            } else {
              setPriority('');
            }
          }}
          placeholder="Enter a number between 1 and 5"
          keyboardType="numeric"
          colors={{
            backgroundColor: Colors.SECONDARY_BACKGROUND,
            textColor: Colors.PRIMARY_TEXT,
            placeholderTextColor: Colors.SECONDARY_TEXT,
            neutralBorderColor: Colors.DIVIDER,
          }}
          style={{ marginBottom: 16 }}
        />

        {/* Subtasks Section */}
        <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT, marginBottom: 8 }}>
          Subtasks:
        </CustomText>
        <StyledInput
          mode="outlined"
          labelText="New Subtask"
          value={newSubtask}
          onChangeText={setNewSubtask}
          placeholder="Enter subtask title"
          keyboardType="default"
          colors={{
            backgroundColor: Colors.SECONDARY_BACKGROUND,
            textColor: Colors.PRIMARY_TEXT,
            placeholderTextColor: Colors.SECONDARY_TEXT,
            neutralBorderColor: Colors.DIVIDER,
          }}
          style={{ marginBottom: 8 }}
        />
        <CustomButton
          title="Add Subtask"
          onPress={addSubtask}
          style={{ backgroundColor: Colors.ACCENT, paddingVertical: 8, borderRadius: 6, marginBottom: 16 }}
          textStyle={{ color: Colors.PRIMARY_TEXT, textAlign: 'center', fontWeight: 'bold' }}
        />
        {subtasks.length > 0 && (
          <View style={{ marginBottom: 16 }}>
            {subtasks.map((subtask, index) => (
              <View key={index} style={{ flexDirection: 'row', alignItems: 'center', marginVertical: 4 }}>
                <CustomText variant="headingSmall" style={{ color: Colors.PRIMARY_TEXT, flex: 1 }}>
                  {subtask}
                </CustomText>
                <TouchableOpacity onPress={() => setSubtasks((prev) => prev.filter((_, i) => i !== index))}>
                  <MaterialIcons name="delete" size={20} color={Colors.ERROR} />
                </TouchableOpacity>
              </View>
            ))}
          </View>
        )}

        {/* Attachments Section */}
        <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT, marginBottom: 8 }}>
          Attachments:
        </CustomText>
        <CustomButton
          title="Add Attachment"
          onPress={handleAddNewAttachment}
          style={{ backgroundColor: Colors.ACCENT, paddingVertical: 8, borderRadius: 6, marginBottom: 16 }}
          textStyle={{ color: Colors.PRIMARY_TEXT, textAlign: 'center', fontWeight: 'bold' }}
        />
        {attachments.length > 0 && (
          <View style={{ marginBottom: 16 }}>
            {attachments.map((att, index) => (
              <View key={index} style={{ flexDirection: 'row', alignItems: 'center', marginVertical: 4 }}>
                <CustomText variant="headingSmall" style={{ color: Colors.PRIMARY_TEXT, flex: 1 }}>
                  {att.name}
                </CustomText>
                <TouchableOpacity onPress={() => setAttachments((prev) => prev.filter((_, i) => i !== index))}>
                  <MaterialIcons name="delete" size={20} color={Colors.ERROR} />
                </TouchableOpacity>
              </View>
            ))}
          </View>
        )}
      </ScrollView>

      {/* Sticky Create Task Button */}
      <CustomButton
        title={isMutating ? 'Creating...' : 'Create Task'}
        onPress={handleCreateTask}
        style={{
          backgroundColor: Colors.BUTTON,
          position: 'absolute',
          bottom: 24,
          left: 24,
          right: 24,
          paddingVertical: 12,
          borderRadius: 8,
        }}
        textStyle={{
          color: Colors.PRIMARY_TEXT,
          textAlign: 'center',
          fontWeight: 'bold',
          fontSize: 20,
        }}
      />

      {/* Updated DatePicker using the latest component API (isVisible instead of visible) */}
      {datePickerVisible && (
        <CustomDatePicker
          visible={datePickerVisible}
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

      {reminderPickerVisible && (
        <CustomDatePicker
          visible={reminderPickerVisible}
          date={reminderAt || new Date()}
          mode="date"
          minimumDate={new Date()}
          onConfirm={(date: Date) => {
            setReminderAt(date);
            setReminderPickerVisible(false);
          }}
          onCancel={() => setReminderPickerVisible(false)}
        />
      )}
    </SafeAreaView>
  );
}
