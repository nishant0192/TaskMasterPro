// ConfirmationModal.tsx
import React, { useEffect, useState } from 'react';
import { Modal, View, Text, TouchableOpacity, Animated } from 'react-native';
import { AntDesign } from '@expo/vector-icons';
import Colors from '@/constants/Colors';

interface ConfirmationModalProps {
    visible: boolean;
    title?: string;
    message?: string;
    onConfirm: () => void;
    onCancel: () => void;
}

const ConfirmationModal: React.FC<ConfirmationModalProps> = ({
    visible,
    title,
    message,
    onConfirm,
    onCancel,
}) => {
    const [fadeAnim] = useState(new Animated.Value(0));

    useEffect(() => {
        if (visible) {
            Animated.timing(fadeAnim, {
                toValue: 1,
                duration: 300,
                useNativeDriver: true,
            }).start();
        } else {
            fadeAnim.setValue(0);
        }
    }, [visible, fadeAnim]);

    return (
        <View>
            <Modal transparent visible={visible} animationType="fade" onRequestClose={onCancel}>
                <Animated.View style={{ opacity: fadeAnim }} className="flex-1 bg-black bg-opacity-70 justify-center items-center">
                    <View
                        className="p-5 rounded-lg w-4/5"
                        style={{ backgroundColor: Colors.SECONDARY_BACKGROUND }}
                    >
                        {title && (
                            <Text className="text-lg text-center mb-2" style={{ color: Colors.PRIMARY_TEXT }}>
                                {title}
                            </Text>
                        )}
                        {message && (
                            <Text className="text-base text-center mb-5" style={{ color: Colors.PRIMARY_TEXT }}>
                                {message}
                            </Text>
                        )}
                        <View className="flex-row justify-around">
                            <TouchableOpacity
                                onPress={onCancel}
                                className="py-2 px-5 w-32 rounded"  // Adjusted width to w-32 (larger size)
                                style={{ backgroundColor: Colors.DIVIDER }}
                            >
                                <AntDesign name="close" size={20} color={Colors.PRIMARY_TEXT} />
                            </TouchableOpacity>
                            <TouchableOpacity
                                onPress={onConfirm}
                                className="py-2 px-5 w-32 rounded"  // Adjusted width to w-32 (larger size)
                                style={{ backgroundColor: Colors.ERROR }}
                            >
                                <AntDesign name="check" size={20} color={Colors.PRIMARY_TEXT} />
                            </TouchableOpacity>
                        </View>
                    </View>
                </Animated.View>
            </Modal>
        </View>
    );
};

export default ConfirmationModal;
