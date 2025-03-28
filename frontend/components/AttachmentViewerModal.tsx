import React from 'react';
import { Modal, TouchableOpacity, View, Image, StyleSheet, Dimensions } from 'react-native';
import { WebView } from 'react-native-webview';
import CustomText from '@/components/CustomText';
import CustomButton from '@/components/CustomButton';
import { Colors } from '@/constants/Colors';

const { width, height } = Dimensions.get('window');

interface AttachmentViewerModalProps {
  visible: boolean;
  attachment: {
    fileUrl: string;
    fileType?: string;
    fileName?: string;
    name?: string;
  } | null;
  onClose: () => void;
}

const AttachmentViewerModal: React.FC<AttachmentViewerModalProps> = ({ visible, attachment, onClose }) => {
  if (!attachment) return null;

  const imageRegex = /\.(jpg|jpeg|png|gif)$/i;
  const pdfRegex = /\.(pdf)$/i;
  const isImage =
    (attachment.fileType && attachment.fileType.startsWith('image/')) ||
    (attachment.fileUrl && imageRegex.test(attachment.fileUrl));
  const isPDF =
    (attachment.fileType && attachment.fileType === 'application/pdf') ||
    (attachment.fileUrl && pdfRegex.test(attachment.fileUrl));

  return (
    <View>
      <Modal visible={visible} transparent animationType="fade" onRequestClose={onClose}>
        <TouchableOpacity style={styles.overlay} activeOpacity={1} onPress={onClose}>
          <View style={styles.modalContainer}>
            {isImage ? (
              <Image source={{ uri: attachment.fileUrl }} style={styles.image} resizeMode="contain" />
            ) : isPDF ? (
              <WebView source={{ uri: attachment.fileUrl }} style={styles.pdf} />
            ) : (
              <CustomText variant="headingMedium" style={{ color: Colors.PRIMARY_TEXT }}>
                Unsupported file type.
              </CustomText>
            )}
            <CustomButton
              title="Close"
              onPress={onClose}
              style={styles.closeButton}
              textStyle={styles.closeButtonText}
            />
          </View>
        </TouchableOpacity>
      </Modal>
    </View>
  );
};

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.8)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContainer: {
    width: width * 0.9,
    height: height * 0.8,
    backgroundColor: Colors.PRIMARY_BACKGROUND,
    borderRadius: 8,
    padding: 16,
    alignItems: 'center',
  },
  image: {
    width: '100%',
    height: '80%',
  },
  pdf: {
    width: '100%',
    height: '80%',
  },
  closeButton: {
    marginTop: 16,
    paddingVertical: 8,
    paddingHorizontal: 16,
    backgroundColor: Colors.BUTTON,
    borderRadius: 4,
  },
  closeButtonText: {
    color: Colors.PRIMARY_TEXT,
    textAlign: 'center',
  },
});

export default AttachmentViewerModal;
