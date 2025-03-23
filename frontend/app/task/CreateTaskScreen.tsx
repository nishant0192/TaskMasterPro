import React, { useState } from 'react';
import { SafeAreaView, Alert, View, TouchableOpacity, ScrollView } from 'react-native';
import CustomText from '@/components/CustomText';
import CustomButton from '@/components/CustomButton';
import StyledInput from '@/components/StyledInput';
import CustomDatePicker from '@/components/CustomDatePicker';
import { useCreateTask } from '@/api/task/useTask';
import { useRouter } from 'expo-router';
import { Colors } from '@/constants/Colors';
import { AntDesign } from '@expo/vector-icons';

export default function CreateTaskScreen() {
    const [title, setTitle] = useState('');
    const [description, setDescription] = useState('');
    const [priority, setPriority] = useState('');
    const [dueDate, setDueDate] = useState<Date | undefined>(undefined);
    const [datePickerVisible, setDatePickerVisible] = useState(false);

    const router = useRouter();
    const { mutate: createTask, status, error } = useCreateTask();
    const isMutating = status === 'pending';

    const handleCreateTask = () => {
        if (!title) {
            Alert.alert('Error', 'Title is required');
            return;
        }
        createTask(
            {
                title,
                description,
                dueDate: dueDate ? dueDate.toISOString() : undefined,
                priority: priority ? Number(priority) : undefined,
            },
            {
                onSuccess: (data) => {
                    console.log('Task created successfully', data);
                    router.push('/task/TaskListScreen');
                },
                onError: (err) => {
                    console.error('Error creating task', err);
                    Alert.alert('Error', 'Failed to create task');
                },
            }
        );
    };

    // Use the color palette from Colors for inputs.
    const inputColors = {
        backgroundColor: Colors.SECONDARY_BACKGROUND, // Dark Charcoal
        textColor: Colors.PRIMARY_TEXT,               // Light Gray
        placeholderTextColor: Colors.SECONDARY_TEXT,  // Muted Gray
        neutralBorderColor: Colors.DIVIDER,           // Divider
        successBorderColor: Colors.SUCCESS,           // Success green
        errorBorderColor: Colors.ERROR,               // Error red
        clearIconColor: Colors.INACTIVE,              // Dim Gray
    };

    return (
        <SafeAreaView style={{ flex: 1, backgroundColor: Colors.PRIMARY_BACKGROUND, padding: 24 }}>
            <ScrollView
                contentContainerStyle={{ paddingBottom: 40 }}
                keyboardShouldPersistTaps="handled"
                showsVerticalScrollIndicator={false}
            >
                <View>
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
                        colors={inputColors}
                    />

                    <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT, marginTop: 16 }}>
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
                        colors={inputColors}
                        numberOfLines={10}
                    />

                    <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT, marginTop: 16, marginBottom: 8 }}>
                        Due Date:
                    </CustomText>
                    <TouchableOpacity
                        onPress={() => setDatePickerVisible(true)}
                        style={{
                            marginTop: 8,
                            paddingVertical: 12,
                            paddingHorizontal: 16,
                            borderWidth: 1,
                            borderColor: Colors.DIVIDER,
                            borderRadius: 8,
                            justifyContent: 'center',
                            backgroundColor: Colors.SECONDARY_BACKGROUND,
                            flexDirection: 'row',  // Align children (text and icon) in a row
                            alignItems: 'center',  // Vertically center the items
                        }}
                    >
                        <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT, paddingLeft: 4, paddingTop: 4, flex: 1 }}>
                            {dueDate ? dueDate.toLocaleDateString() : 'Select Due Date'}
                        </CustomText>

                        {/* Calendar Icon */}
                        <AntDesign name="calendar" size={24} color={Colors.PRIMARY_TEXT} />
                    </TouchableOpacity>

                    <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT, marginTop: 25 }}>
                        Priority:
                    </CustomText>
                    <StyledInput
                        mode="outlined"
                        labelText="Priority (1-5)"
                        value={priority}
                        onChangeText={(text) => {
                            // Remove non-digit characters.
                            let numericText = text.replace(/[^0-9]/g, "");

                            // Ensure the numeric text is a valid number between 1 and 5.
                            const num = parseInt(numericText, 10);

                            if (!isNaN(num)) {
                                // If the number is between 1 and 5, update the state.
                                if (num >= 1 && num <= 5) {
                                    setPriority(numericText);
                                } else {
                                    // If the number is outside the range, set to "5" or "1" depending on the limit.
                                    setPriority(num > 5 ? '5' : '1');
                                }
                            } else {
                                setPriority(""); // If not a valid number, clear the input.
                            }
                        }}
                        placeholder="Enter a number between 1 and 5"
                        keyboardType="numeric"
                        colors={inputColors}
                        style={{ marginTop: 16 }}
                    />


                </View>
                {error && (
                    <CustomText variant="headingMedium" style={{ color: Colors.ERROR, textAlign: 'center', marginTop: 16 }}>
                        {error.message}
                    </CustomText>
                )}
                <CustomButton
                    title={isMutating ? 'Creating...' : 'Create Task'}
                    onPress={handleCreateTask}
                    style={{
                        backgroundColor: Colors.BUTTON,
                        marginTop: 32,
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
                {datePickerVisible && (
                    <CustomDatePicker
                        date={dueDate || new Date()}
                        mode="date"
                        minimumDate={new Date()} // Due date must be in the future.
                        onConfirm={(date: Date) => {
                            setDueDate(date);
                            setDatePickerVisible(false);
                        }}
                        onCancel={() => setDatePickerVisible(false)}
                    />
                )}
            </ScrollView>
        </SafeAreaView>
    );
}
