import React, { useState, useRef, useEffect } from 'react';
import {
    View,
    Modal,
    Button,
    StyleSheet,
    Animated,
    Platform,
} from 'react-native';
import DateTimePicker from '@react-native-community/datetimepicker';

interface CustomDatePickerProps {
    date: Date;
    onConfirm: (date: Date) => void;
    onCancel: () => void;
    mode?: 'date' | 'time' | 'datetime';
    minimumDate?: Date;
}

const CustomDatePicker: React.FC<CustomDatePickerProps> = ({
    date,
    onConfirm,
    onCancel,
    mode = 'date',
    minimumDate,
}) => {
    const [selectedDate, setSelectedDate] = useState(date);
    const fadeAnim = useRef(new Animated.Value(0)).current;

    useEffect(() => {
        Animated.timing(fadeAnim, {
            toValue: 1,
            duration: 300,
            useNativeDriver: true,
        }).start();
    }, [fadeAnim]);

    const handleChange = (event: any, newDate?: Date) => {
        if (Platform.OS === 'android') {
            if (newDate) {
                // Validate that newDate is not in the past.
                if (minimumDate && newDate < minimumDate) {
                    onCancel();
                } else {
                    onConfirm(newDate);
                }
            } else {
                onCancel();
            }
        } else {
            if (newDate) {
                setSelectedDate(newDate);
            }
        }
    };

    const handleIOSConfirm = () => {
        // On iOS, check that the selected date meets the minimumDate condition.
        if (minimumDate && selectedDate < minimumDate) {
            onCancel();
        } else {
            onConfirm(selectedDate);
        }
    };

    return (
        <Modal transparent animationType="fade" visible>
            <Animated.View style={[styles.overlay, { opacity: fadeAnim }]}>
                <View style={styles.container}>
                    <DateTimePicker
                        value={selectedDate}
                        mode={mode}
                        display={Platform.OS === 'ios' ? 'spinner' : 'default'}
                        onChange={handleChange}
                        minimumDate={minimumDate}
                        style={{ width: '100%' }}
                    />
                    {Platform.OS === 'ios' && (
                        <View style={styles.buttonRow}>
                            <Button title="Cancel" onPress={onCancel} color="#FF4500" />
                            <Button title="Confirm" onPress={handleIOSConfirm} color="#32CD32" />
                        </View>
                    )}
                </View>
            </Animated.View>
        </Modal>
    );
};

const styles = StyleSheet.create({
    overlay: {
        flex: 1,
        backgroundColor: 'rgba(0,0,0,0.5)',
        justifyContent: 'center',
        alignItems: 'center',
    },
    container: {
        backgroundColor: '#1F2937',
        borderRadius: 10,
        padding: 20,
        width: '80%',
    },
    buttonRow: {
        flexDirection: 'row',
        justifyContent: 'space-around',
        marginTop: 20,
    },
});

export default CustomDatePicker;
