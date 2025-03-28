// TaskDetailsScreen.tsx
import React, { useState, useEffect } from 'react';
import {
  SafeAreaView,
  ScrollView,
  View,
  Alert,
  TouchableOpacity,
  ActivityIndicator,
  Modal,
  StyleSheet,
  Dimensions,
} from 'react-native';
import { useRouter, useLocalSearchParams } from 'expo-router';
import CustomText from '@/components/CustomText';
import StyledInput from '@/components/StyledInput';
import CustomButton from '@/components/CustomButton';
import CustomDatePicker from '@/components/CustomDatePicker';
import ConfirmationModal from '@/components/ConfirmationModal';
import AttachmentViewerModal from '@/components/AttachmentViewerModal';
import DraggableFlatList, { RenderItemParams } from 'react-native-draggable-flatlist';
import {
  useGetTaskById,
  useUpdateTask,
  useCreateAttachment,
  useGetAttachments,
  useGetSubtasks,
  useCreateSubtask,
  useUpdateSubtask,
  useDeleteSubtask,
  useDeleteAttachment,
} from '@/api/task/useTask';
import { Colors } from '@/constants/Colors';
import { AntDesign } from '@expo/vector-icons';
import { useSharedValue } from 'react-native-reanimated';
import { Slider as AwesomeSlider } from 'react-native-awesome-slider';
import * as Haptics from 'expo-haptics';
import * as DocumentPicker from 'expo-document-picker';
import * as FileSystem from 'expo-file-system';

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
  const router = useRouter();

  // API hooks / mutations.
  const getTaskByIdMutation = useGetTaskById(id);
  const updateTaskMutation = useUpdateTask();
  const getAttachmentsMutation = useGetAttachments(id);
  const getSubtasksMutation = useGetSubtasks(id);
  const createAttachmentMutation = useCreateAttachment();
  const createSubtaskMutation = useCreateSubtask();
  const updateSubtaskMutation = useUpdateSubtask();
  const deleteSubtaskMutation = useDeleteSubtask();
  const deleteAttachmentMutation = useDeleteAttachment();

  // Task details state.
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [dueDate, setDueDate] = useState<Date | undefined>(undefined);
  const [priority, setPriority] = useState('');
  const [status, setStatus] = useState('');
  const [progress, setProgress] = useState('0');
  const [isArchived, setIsArchived] = useState<boolean>(false);
  const [datePickerVisible, setDatePickerVisible] = useState(false);
  const [statusDropdownVisible, setStatusDropdownVisible] = useState(false);

  // Attachments and subtasks state.
  const [attachments, setAttachments] = useState<any[]>([]);
  const [subtasks, setSubtasks] = useState<any[]>([]);
  // For editing a subtask (via modal).
  const [editingSubtask, setEditingSubtask] = useState<any>(null);
  const [editSubtaskText, setEditSubtaskText] = useState('');
  const [editSubtaskReminder, setEditSubtaskReminder] = useState<Date | undefined>(undefined);
  const [showEditSubtaskDatePicker, setShowEditSubtaskDatePicker] = useState(false);
  // For adding a new subtask via modal.
  const [showAddSubtaskModal, setShowAddSubtaskModal] = useState(false);
  const [newSubtask, setNewSubtask] = useState('');
  const [newSubtaskReminder, setNewSubtaskReminder] = useState<Date | undefined>(undefined);
  const [showNewSubtaskDatePicker, setShowNewSubtaskDatePicker] = useState(false);

  // Processing state.
  const [isProcessing, setIsProcessing] = useState(false);

  // Attachment viewer modal state.
  const [showAttachmentModal, setShowAttachmentModal] = useState(false);
  const [selectedAttachment, setSelectedAttachment] = useState<{
    fileType?: string;
    fileUrl: string;
    fileName?: string;
    name?: string;
  } | null>(null);

  // Confirmation modal state for deletions.
  const [confirmationModalVisible, setConfirmationModalVisible] = useState(false);
  const [deleteType, setDeleteType] = useState<'subtask' | 'attachment' | null>(null);
  const [deleteId, setDeleteId] = useState<string | null>(null);

  // Shared values for the progress slider.
  const progressShared = useSharedValue(Number(progress));
  const minimumValue = useSharedValue(0);
  const maximumValue = useSharedValue(100);

  // Fetch data on mount or when id changes.
  useEffect(() => {
    if (id) {
      getTaskByIdMutation.mutate();
      getAttachmentsMutation.mutate();
      getSubtasksMutation.mutate();
    }
  }, [id]);

  // Populate task details.
  useEffect(() => {
    if (getTaskByIdMutation.data?.task) {
      const task = getTaskByIdMutation.data.task;
      setTitle(task.title);
      setDescription(task.description);
      setDueDate(task.dueDate ? new Date(task.dueDate) : undefined);
      setPriority(task.priority ? String(task.priority) : '');
      setStatus(task.status === 'IN_PROGRESS' ? 'in_progres' : task.status || '');
      setProgress(task.progress ? String(task.progress) : '0');
      setIsArchived(task.isArchived);
      progressShared.value = task.progress ? Number(task.progress) : 0;
    }
  }, [getTaskByIdMutation.data]);

  // Populate attachments.
  useEffect(() => {
    if (getAttachmentsMutation.data?.attachments) {
      setAttachments(getAttachmentsMutation.data.attachments);
    }
  }, [getAttachmentsMutation.data]);

  // Populate subtasks.
  useEffect(() => {
    if (getSubtasksMutation.data?.subtasks) {
      setSubtasks(getSubtasksMutation.data.subtasks);
    }
  }, [getSubtasksMutation.data]);

  // Ensure slider shows 100% when status is DONE.
  useEffect(() => {
    if (status === 'DONE' && Number(progress) !== 100) {
      setProgress('100');
      progressShared.value = 100;
    }
  }, [status]);

  // Update task handler.
  const handleUpdateTask = () => {
    if (!title) {
      Alert.alert('Error', 'Title is required');
      return;
    }
    if (isProcessing) return;
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

  // Attachment handler.
  const handleAddAttachment = async () => {
    try {
      setIsProcessing(true);
      const result = await DocumentPicker.getDocumentAsync({});
      if (result.canceled) {
        setIsProcessing(false);
        return;
      }
      if (result.assets && result.assets[0]) {
        const asset = result.assets[0];
        const fileData = await FileSystem.readAsStringAsync(asset.uri, {
          encoding: FileSystem.EncodingType.Base64,
        });
        createAttachmentMutation.mutate(
          {
            taskId: id,
            fileName: asset.name,
            fileData,
          },
          {
            onSuccess: (data) => {
              setAttachments((prev) => [...prev, data.attachment]);
              setIsProcessing(false);
            },
            onError: (err) => {
              console.error('Error adding attachment', err);
              Alert.alert('Error', 'Failed to add attachment');
              setIsProcessing(false);
            },
          }
        );
      }
    } catch (err) {
      console.error('Attachment error:', err);
      Alert.alert('Error', 'Failed to add attachment');
      setIsProcessing(false);
    }
  };

  // Subtask creation (via modal).
  const handleAddSubtask = () => {
    if (!newSubtask.trim()) {
      Alert.alert('Error', 'Subtask title cannot be empty');
      return;
    }
    setIsProcessing(true);
    const order = subtasks.length + 1;
    createSubtaskMutation.mutate(
      {
        taskId: id,
        title: newSubtask,
        order,
        reminderAt: newSubtaskReminder ? newSubtaskReminder.toISOString() : undefined,
      },
      {
        onSuccess: (data) => {
          setSubtasks((prev) => [...prev, data.subtask]);
          setNewSubtask('');
          setNewSubtaskReminder(undefined);
          setIsProcessing(false);
        },
        onError: (err) => {
          console.error('Error creating subtask', err);
          Alert.alert('Error', 'Failed to add subtask');
          setIsProcessing(false);
        },
      }
    );
  };

  // Subtask deletion.
  const handleDeleteSubtask = (subtaskId: string) => {
    setIsProcessing(true);
    deleteSubtaskMutation.mutate(
      { id: subtaskId },
      {
        onSuccess: () => {
          setSubtasks((prev) => prev.filter((s: any) => s.id !== subtaskId));
          setIsProcessing(false);
        },
        onError: (err) => {
          console.error('Error deleting subtask', err);
          Alert.alert('Error', 'Failed to delete subtask');
          setIsProcessing(false);
        },
      }
    );
  };

  // Attachment deletion.
  const handleDeleteAttachment = (attachmentId: string) => {
    setIsProcessing(true);
    deleteAttachmentMutation.mutate(
      { id: attachmentId },
      {
        onSuccess: () => {
          setAttachments((prev) => prev.filter((att) => att.id !== attachmentId));
          setIsProcessing(false);
        },
        onError: (err) => {
          console.error('Error deleting attachment', err);
          Alert.alert('Error', 'Failed to delete attachment');
          setIsProcessing(false);
        },
      }
    );
  };

  // Open the edit modal for a subtask.
  const openEditSubtaskModal = (subtask: any) => {
    setEditingSubtask(subtask);
    setEditSubtaskText(subtask.title);
    setEditSubtaskReminder(subtask.reminderAt ? new Date(subtask.reminderAt) : undefined);
  };

  // Update subtask handler (from the edit modal).
  const handleUpdateSubtask = () => {
    if (!editSubtaskText.trim()) {
      Alert.alert('Error', 'Subtask title cannot be empty');
      return;
    }
    setIsProcessing(true);
    updateSubtaskMutation.mutate(
      {
        id: editingSubtask.id,
        title: editSubtaskText,
        reminderAt: editSubtaskReminder ? editSubtaskReminder.toISOString() : undefined,
      },
      {
        onSuccess: () => {
          setSubtasks((prev) =>
            prev.map((s: any) =>
              s.id === editingSubtask.id
                ? { ...s, title: editSubtaskText, reminderAt: editSubtaskReminder ? editSubtaskReminder.toISOString() : s.reminderAt }
                : s
            )
          );
          setEditingSubtask(null);
          setEditSubtaskText('');
          setEditSubtaskReminder(undefined);
          setIsProcessing(false);
        },
        onError: (err) => {
          console.error('Error updating subtask', err);
          Alert.alert('Error', 'Failed to update subtask');
          setIsProcessing(false);
        },
      }
    );
  };

  // Handle attachment view.
  const handleViewAttachment = (attachment: {
    fileType?: string;
    fileUrl: string;
    fileName?: string;
    name?: string;
  }): void => {
    const imageRegex = /\.(jpg|jpeg|png|gif)$/i;
    const pdfRegex = /\.(pdf)$/i;
    if (
      (attachment.fileType && attachment.fileType.startsWith('image/')) ||
      (attachment.fileUrl && imageRegex.test(attachment.fileUrl))
    ) {
      setSelectedAttachment(attachment);
      setShowAttachmentModal(true);
    } else if (
      (attachment.fileType && attachment.fileType === 'application/pdf') ||
      (attachment.fileUrl && pdfRegex.test(attachment.fileUrl))
    ) {
      setSelectedAttachment(attachment);
      setShowAttachmentModal(true);
    } else {
      Alert.alert('View Attachment', `File: ${attachment.fileName || attachment.name}`);
    }
  };

  // Update subtask order after dragging.
  const handleSubtasksDragEnd = ({ data }: { data: any[] }) => {
    setSubtasks(data);
    data.forEach((sub, index) => {
      if (sub.order !== index + 1) {
        updateSubtaskMutation.mutate({ id: sub.id, order: index + 1 });
      }
    });
  };

  // Render each subtask row.
  const renderSubtaskItem = ({ item, drag, isActive, getIndex }: RenderItemParams<any>) => (
    <View style={styles.subtaskRow}>
      <TouchableOpacity onLongPress={drag} style={{ marginRight: 8 }}>
        <AntDesign name="bars" size={20} color={Colors.PRIMARY_TEXT} />
      </TouchableOpacity>
      <CustomText variant="headingSmall" style={{ color: Colors.PRIMARY_TEXT, flex: 1 }}>
        {(getIndex?.() ?? 0) + 1}. {item.title}
      </CustomText>
      {item.reminderAt && (
        <CustomText variant="headingSmall" style={{ color: Colors.SECONDARY_TEXT, marginRight: 8 }}>
          {new Date(item.reminderAt).toLocaleString()}
        </CustomText>
      )}
      <TouchableOpacity onPress={() => openEditSubtaskModal(item)} style={{ marginHorizontal: 8 }}>
        <AntDesign name="edit" size={24} color={Colors.BUTTON} />
      </TouchableOpacity>
      <TouchableOpacity
        onPress={() => {
          setDeleteType('subtask');
          setDeleteId(item.id);
          setConfirmationModalVisible(true);
        }}
        style={{ marginHorizontal: 8 }}
      >
        <AntDesign name="delete" size={24} color={Colors.ERROR} />
      </TouchableOpacity>
    </View>
  );

  // Confirmation modal message based on delete type.
  const confirmationMessage =
    deleteType === 'subtask'
      ? 'Are you sure you want to delete this subtask?'
      : deleteType === 'attachment'
      ? 'Are you sure you want to delete this attachment?'
      : '';

  if (getTaskByIdMutation.isPending) {
    return (
      <SafeAreaView style={styles.centeredContainer}>
        <ActivityIndicator size="large" color={Colors.BUTTON} />
        <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT, marginTop: 16 }}>
          Loading task details...
        </CustomText>
      </SafeAreaView>
    );
  }

  if (getTaskByIdMutation.error) {
    return (
      <SafeAreaView style={styles.centeredContainer}>
        <CustomText variant="headingMedium" style={{ color: Colors.ERROR, textAlign: 'center' }}>
          Error fetching task details: {getTaskByIdMutation.error.message}
        </CustomText>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={{ flexGrow: 1, paddingBottom: 200 }} showsVerticalScrollIndicator={false}>
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
            style={styles.inputTouchable}
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
            style={styles.inputTouchable}
          >
            <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT, flex: 1 }}>
              {status ? (statusOptions.find((option) => option.value === status)?.label || status) : 'Select Status'}
            </CustomText>
            <AntDesign name="caretdown" size={18} color={Colors.PRIMARY_TEXT} />
          </TouchableOpacity>
          {statusDropdownVisible && (
            <View style={styles.dropdownContainer}>
              {statusOptions.map((option) => (
                <TouchableOpacity
                  key={option.value}
                  onPress={() => {
                    setStatus(option.value);
                    setStatusDropdownVisible(false);
                  }}
                  style={styles.dropdownItem}
                >
                  <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT }}>
                    {option.label}
                  </CustomText>
                </TouchableOpacity>
              ))}
            </View>
          )}
        </View>

        {/* Progress Slider */}
        <View style={{ marginBottom: 24 }}>
          <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT, marginBottom: 8 }}>
            Progress (%):
          </CustomText>
          <AwesomeSlider
            style={{ height: 10 }}
            containerStyle={{ borderRadius: 8 }}
            progress={progressShared}
            minimumValue={minimumValue}
            maximumValue={maximumValue}
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
            {motivatingSentences[Number(progress)] || ''}
          </CustomText>
        </View>

        {/* Attachments Section */}
        <View style={{ marginBottom: 24 }}>
          <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT, marginBottom: 8 }}>
            Attachments:
          </CustomText>
          {attachments.length > 0 ? (
            attachments.map((att, index) => (
              <View key={index} style={styles.attachmentRow}>
                <CustomText variant="headingSmall" style={{ color: Colors.PRIMARY_TEXT, flex: 1 }}>
                  {att.fileName || att.name}
                </CustomText>
                <TouchableOpacity onPress={() => handleViewAttachment(att)} style={{ marginHorizontal: 8 }}>
                  <AntDesign name="eye" size={24} color={Colors.BUTTON} />
                </TouchableOpacity>
                <TouchableOpacity
                  onPress={() => {
                    setDeleteType('attachment');
                    setDeleteId(att.id);
                    setConfirmationModalVisible(true);
                  }}
                  style={{ marginHorizontal: 8 }}
                >
                  <AntDesign name="delete" size={24} color={Colors.ERROR} />
                </TouchableOpacity>
              </View>
            ))
          ) : (
            <CustomText variant="headingSmall" style={{ color: Colors.SECONDARY_TEXT }}>
              No attachments available.
            </CustomText>
          )}
          <CustomButton
            title="Add Attachment"
            onPress={handleAddAttachment}
            style={{ backgroundColor: Colors.ACCENT, paddingVertical: 8, borderRadius: 6, marginTop: 8 }}
            textStyle={{ color: Colors.PRIMARY_TEXT, textAlign: 'center', fontWeight: 'bold' }}
          />
        </View>

        {/* Subtasks Section */}
        <View style={{ marginBottom: 24 }}>
          <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT, marginBottom: 8 }}>
            Subtasks:
          </CustomText>
          <DraggableFlatList
            data={subtasks}
            keyExtractor={(item) => item.id}
            renderItem={renderSubtaskItem}
            onDragEnd={({ data }) => handleSubtasksDragEnd({ data })}
            containerStyle={{ maxHeight: 300 }}
          />
          <CustomButton
            title="Add Subtask"
            onPress={() => setShowAddSubtaskModal(true)}
            style={{ backgroundColor: Colors.ACCENT, paddingVertical: 8, borderRadius: 6, marginTop: 16 }}
            textStyle={{ color: Colors.PRIMARY_TEXT, textAlign: 'center', fontWeight: 'bold' }}
          />
        </View>
      </ScrollView>

      {/* Sticky Update Task Button */}
      <View style={styles.stickyButtonContainer}>
        <CustomButton
          title="Update Task"
          onPress={handleUpdateTask}
          disabled={isProcessing}
          style={styles.updateTaskButton}
          textStyle={styles.updateTaskButtonText}
          icon={<AntDesign name="checkcircleo" size={24} color={Colors.PRIMARY_TEXT} style={{ marginRight: 8 }} />}
        />
      </View>

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

      <AttachmentViewerModal
        visible={showAttachmentModal}
        attachment={selectedAttachment}
        onClose={() => setShowAttachmentModal(false)}
      />

      {showEditSubtaskDatePicker && (
        <CustomDatePicker
          visible={showEditSubtaskDatePicker}
          date={editSubtaskReminder || new Date()}
          mode="datetime"
          minimumDate={new Date()}
          onConfirm={(date: Date) => {
            setEditSubtaskReminder(date);
            setShowEditSubtaskDatePicker(false);
          }}
          onCancel={() => setShowEditSubtaskDatePicker(false)}
        />
      )}

      {confirmationModalVisible && deleteId && (
        <ConfirmationModal
          visible={confirmationModalVisible}
          title={deleteType === 'subtask' ? 'Delete Subtask' : 'Delete Attachment'}
          message={confirmationMessage}
          onConfirm={() => {
            if (deleteType === 'subtask') {
              handleDeleteSubtask(deleteId);
            } else if (deleteType === 'attachment') {
              handleDeleteAttachment(deleteId);
            }
            setConfirmationModalVisible(false);
            setDeleteType(null);
            setDeleteId(null);
          }}
          onCancel={() => {
            setConfirmationModalVisible(false);
            setDeleteType(null);
            setDeleteId(null);
          }}
        />
      )}

      
      {/* Edit Subtask Modal */}
      <View>
      {editingSubtask && (
        <Modal transparent animationType="fade" visible={!!editingSubtask} onRequestClose={() => setEditingSubtask(null)}>
          <View style={styles.editModalOverlay}>
            <View style={styles.editModalContainer}>
              <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT, marginBottom: 10 }}>
                Edit Subtask
              </CustomText>
              <StyledInput
                mode="outlined"
                labelText="Subtask Title"
                value={editSubtaskText}
                onChangeText={setEditSubtaskText}
                placeholder="Edit subtask"
                keyboardType="default"
                colors={{
                  backgroundColor: Colors.SECONDARY_BACKGROUND,
                  textColor: Colors.PRIMARY_TEXT,
                  placeholderTextColor: Colors.SECONDARY_TEXT,
                  neutralBorderColor: Colors.DIVIDER,
                }}
                style={{ marginBottom: 10 }}
              />
              <TouchableOpacity
                onPress={() => setShowEditSubtaskDatePicker(true)}
                style={styles.inputTouchable}
              >
                <CustomText variant="headingSmall" style={{ color: Colors.PRIMARY_TEXT, flex: 1 }}>
                  {editSubtaskReminder ? `Reminder: ${editSubtaskReminder.toLocaleString()}` : 'Set Reminder'}
                </CustomText>
                <AntDesign name="calendar" size={24} color={Colors.PRIMARY_TEXT} />
              </TouchableOpacity>
              {showEditSubtaskDatePicker && (
                <CustomDatePicker
                  visible={showEditSubtaskDatePicker}
                  date={editSubtaskReminder || new Date()}
                  mode="datetime"
                  minimumDate={new Date()}
                  onConfirm={(date: Date) => {
                    setEditSubtaskReminder(date);
                    setShowEditSubtaskDatePicker(false);
                  }}
                  onCancel={() => setShowEditSubtaskDatePicker(false)}
                />
              )}
              <View style={styles.editModalButtonRow}>
                <CustomButton
                  title="Cancel"
                  onPress={() => {
                    setEditingSubtask(null);
                    setEditSubtaskText('');
                    setEditSubtaskReminder(undefined);
                  }}
                  style={[styles.editModalButton, { backgroundColor: Colors.DIVIDER }]}
                  textStyle={{ color: Colors.PRIMARY_TEXT }}
                />
                <CustomButton
                  title="Update"
                  onPress={handleUpdateSubtask}
                  style={[styles.editModalButton, { backgroundColor: Colors.BUTTON }]}
                  textStyle={{ color: Colors.PRIMARY_TEXT }}
                />
              </View>
            </View>
          </View>
        </Modal>
      )}
      </View>

      {/* Add Subtask Modal */}
      <View>
      {showAddSubtaskModal && (
        <Modal transparent animationType="fade" visible={showAddSubtaskModal} onRequestClose={() => setShowAddSubtaskModal(false)}>
          <View style={styles.editModalOverlay}>
            <View style={styles.editModalContainer}>
              <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT, marginBottom: 10 }}>
                Add Subtask
              </CustomText>
              <StyledInput
                mode="outlined"
                labelText="Subtask Title"
                value={newSubtask}
                onChangeText={setNewSubtask}
                placeholder="Enter subtask"
                keyboardType="default"
                colors={{
                  backgroundColor: Colors.SECONDARY_BACKGROUND,
                  textColor: Colors.PRIMARY_TEXT,
                  placeholderTextColor: Colors.SECONDARY_TEXT,
                  neutralBorderColor: Colors.DIVIDER,
                }}
                style={{ marginBottom: 10 }}
              />
              <TouchableOpacity
                onPress={() => setShowNewSubtaskDatePicker(true)}
                style={styles.inputTouchable}
              >
                <CustomText variant="headingSmall" style={{ color: Colors.PRIMARY_TEXT, flex: 1 }}>
                  {newSubtaskReminder ? `Reminder: ${newSubtaskReminder.toLocaleString()}` : 'Set Reminder'}
                </CustomText>
                <AntDesign name="calendar" size={24} color={Colors.PRIMARY_TEXT} />
              </TouchableOpacity>
              {showNewSubtaskDatePicker && (
                <CustomDatePicker
                  visible={showNewSubtaskDatePicker}
                  date={newSubtaskReminder || new Date()}
                  mode="datetime"
                  minimumDate={new Date()}
                  onConfirm={(date: Date) => {
                    setNewSubtaskReminder(date);
                    setShowNewSubtaskDatePicker(false);
                  }}
                  onCancel={() => setShowNewSubtaskDatePicker(false)}
                />
              )}
              <View style={styles.editModalButtonRow}>
                <CustomButton
                  title="Cancel"
                  onPress={() => {
                    setShowAddSubtaskModal(false);
                    setNewSubtask('');
                    setNewSubtaskReminder(undefined);
                  }}
                  style={[styles.editModalButton, { backgroundColor: Colors.DIVIDER }]}
                  textStyle={{ color: Colors.PRIMARY_TEXT }}
                />
                <CustomButton
                  title="Add"
                  onPress={() => {
                    handleAddSubtask();
                    setShowAddSubtaskModal(false);
                  }}
                  style={[styles.editModalButton, { backgroundColor: Colors.BUTTON }]}
                  textStyle={{ color: Colors.PRIMARY_TEXT }}
                />
              </View>
            </View>
          </View>
        </Modal>
      )}
      </View>
    </SafeAreaView>
  );
}

const { width, height } = Dimensions.get('window');
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.PRIMARY_BACKGROUND,
    padding: 24,
    position: 'relative',
  },
  centeredContainer: {
    flex: 1,
    backgroundColor: Colors.PRIMARY_BACKGROUND,
    padding: 24,
    justifyContent: 'center',
    alignItems: 'center',
  },
  inputTouchable: {
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
  },
  dropdownContainer: {
    marginTop: 8,
    borderWidth: 1,
    borderColor: Colors.DIVIDER,
    borderRadius: 8,
    backgroundColor: Colors.SECONDARY_BACKGROUND,
  },
  dropdownItem: {
    paddingVertical: 12,
    paddingHorizontal: 16,
  },
  attachmentRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 4,
  },
  attachmentAddButton: {
    marginTop: 8,
    alignSelf: 'flex-start',
  },
  subtaskRow: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 8,
    backgroundColor: Colors.SECONDARY_BACKGROUND,
    marginBottom: 4,
    borderRadius: 4,
  },
  stickyButtonContainer: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    borderTopWidth: 1,
    borderTopColor: Colors.DIVIDER,
    paddingVertical: 20,
    paddingHorizontal: 24,
    backgroundColor: Colors.PRIMARY_BACKGROUND,
  },
  updateTaskButton: {
    backgroundColor: Colors.BUTTON,
    paddingVertical: 12,
    borderRadius: 8,
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
  },
  updateTaskButtonText: {
    color: Colors.PRIMARY_TEXT,
    textAlign: 'center',
    fontWeight: 'bold',
    fontSize: 20,
  },
  editModalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.7)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  editModalContainer: {
    width: '90%',
    backgroundColor: Colors.SECONDARY_BACKGROUND,
    padding: 20,
    borderRadius: 10,
  },
  editModalButtonRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginTop: 20,
  },
  editModalButton: {
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 5,
  },
});

