// components/CustomAlert.tsx
import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Modal } from 'react-native';
import {
  AlertNotificationRoot,
  Toast,
  ALERT_TYPE,
} from 'react-native-alert-notification';
import Colors from '@/constants/Colors';
import { Portal } from 'react-native-portalize';
export type AlertType = 'error' | 'success' | 'info' | 'warning';

export interface ShowAlertOptions {
  message: string;
  type?: AlertType;
  title?: string;
}

export interface ShowDialogOptions {
  message: string;
  type?: AlertType;
  title?: string;
  cancelText?: string;
  confirmText?: string;
  onCancelPressed?: () => void;
  onConfirmPressed?: () => void;
}

/**
 * Custom toast renderer to adjust background color.
 * (Width is left to the package's default styling.)
 */
const customToast = (toastOptions: any) => {
  // Use the extra bgColor property passed from showAlert or default to Colors.SURFACE.
  const bgColor = toastOptions?.bgColor || Colors.SURFACE;
  return (
    <View style={[styles.toastContainer, { backgroundColor: bgColor }]}>
      {toastOptions.title ? (
        <Text style={[styles.title, toastOptions.titleStyle]}>
          {toastOptions.title}
        </Text>
      ) : null}
      {toastOptions.textBody ? (
        <Text style={[styles.message, toastOptions.textBodyStyle]}>
          {toastOptions.textBody}
        </Text>
      ) : null}
    </View>
  );
};

/**
 * Displays a toast notification.
 */
export const showAlert = ({
  message,
  type = 'info',
  title,
}: ShowAlertOptions) => {
  let alertType = ALERT_TYPE.INFO;
  let bgColor = Colors.INFO; // default for info

  if (type === 'error') {
    alertType = ALERT_TYPE.DANGER;
    bgColor = Colors.ERROR;
  } else if (type === 'success') {
    alertType = ALERT_TYPE.SUCCESS;
    bgColor = Colors.SUCCESS;
  } else if (type === 'warning') {
    alertType = ALERT_TYPE.WARNING;
    bgColor = Colors.WARNING;
  }

  Toast.show({
    type: alertType,
    title: title || (type.charAt(0).toUpperCase() + type.slice(1)),
    textBody: message,
    titleStyle: { color: Colors.PRIMARY_TEXT },
    textBodyStyle: { color: Colors.PRIMARY_TEXT },
    bgColor: bgColor,
  } as any);
};

// Global modal state
let isModalVisible = false;
let currentModalConfig: {
  title: string;
  message: string;
  cancelText: string;
  confirmText: string;
  onCancel: () => void;
  onConfirm: () => void;
} | null = null;

// Reference to the function that will trigger a re-render of our modal
let updateModalState: ((visible: boolean, config: any) => void) | null = null;

/**
 * Custom Dialog Modal component that will be used instead of the library's Dialog.
 * Its content is wrapped in a full‑screen modalWrapper to ensure correct positioning.
 */
const CustomDialogModal: React.FC = () => {
  const [visible, setVisible] = React.useState(false);
  const [config, setConfig] = React.useState<{
    title: string;
    message: string;
    cancelText: string;
    confirmText: string;
    onCancel: () => void;
    onConfirm: () => void;
  } | null>(null);

  // Set the update function when the component mounts
  React.useEffect(() => {
    updateModalState = (visible: boolean, config: any) => {
      setVisible(visible);
      setConfig(config);
    };

    // Initial state sync
    setVisible(isModalVisible);
    setConfig(currentModalConfig);

    return () => {
      updateModalState = null;
    };
  }, []);

  const handleCancel = () => {
    setVisible(false);
    isModalVisible = false;
    if (config?.onCancel) {
      config.onCancel();
    }
  };

  const handleConfirm = () => {
    setVisible(false);
    isModalVisible = false;
    if (config?.onConfirm) {
      config.onConfirm();
    }
  };

  if (!visible || !config) {
    return null;
  }

  return (
    <Modal
      transparent
      animationType="fade"
      visible={visible}
      onRequestClose={handleCancel}
    >
      <View style={styles.modalWrapper}>
        <View style={styles.modalContainer}>
          <Text style={styles.modalTitle}>{config.title}</Text>
          <Text style={styles.modalMessage}>{config.message}</Text>
          <View style={styles.modalButtonContainer}>
            <TouchableOpacity
              style={styles.modalCancelButton}
              onPress={handleCancel}
            >
              <Text style={styles.modalButtonText}>{config.cancelText}</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.modalConfirmButton}
              onPress={handleConfirm}
            >
              <Text style={styles.modalButtonText}>{config.confirmText}</Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>
    </Modal>
  );
};

/**
 * Displays a dialog (alert) with Cancel and Confirm buttons.
 * Use this for actions like confirming deletion.
 */
export const showDialog = ({
  message,
  type = 'warning',
  title,
  cancelText = 'Cancel',
  confirmText = 'Confirm',
  onCancelPressed,
  onConfirmPressed,
}: ShowDialogOptions) => {
  // Set up the custom dialog modal
  isModalVisible = true;
  currentModalConfig = {
    title: title || (type.charAt(0).toUpperCase() + type.slice(1)),
    message,
    cancelText,
    confirmText,
    onCancel: () => {
      if (onCancelPressed) onCancelPressed();
    },
    onConfirm: () => {
      if (onConfirmPressed) onConfirmPressed();
    },
  };

  // Update the modal if it's already mounted
  if (updateModalState) {
    updateModalState(true, currentModalConfig);
  }
};

/**
 * Custom function to hide the dialog programmatically.
 */
export const hideDialog = () => {
  isModalVisible = false;
  if (updateModalState) {
    updateModalState(false, null);
  }
};

/**
 * CustomAlertRoot wraps your app and applies global configuration
 * for toast notifications and dialogs.
 */
const CustomAlertRoot: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const RootComponent = AlertNotificationRoot as any;
  return (
    <RootComponent
      toastConfig={{
        autoClose: 5000,
        titleStyle: { color: Colors.PRIMARY_TEXT },
        textBodyStyle: { color: Colors.PRIMARY_TEXT },
      }}
      colors={[
        {
          label: Colors.PRIMARY_TEXT,
          card: Colors.SECONDARY_BACKGROUND,
          overlay: Colors.DIVIDER,
          success: Colors.SUCCESS,
          danger: Colors.ERROR,
          warning: Colors.WARNING,
          info: Colors.INFO,
        },
        {
          label: Colors.PRIMARY_TEXT,
          card: Colors.SECONDARY_BACKGROUND,
          overlay: Colors.DIVIDER,
          success: Colors.SUCCESS,
          danger: Colors.ERROR,
          warning: Colors.WARNING,
          info: Colors.INFO,
        },
      ]}
      renderToast={customToast}
    >
      <>
        {children}
        {/* Wrap the custom dialog modal inside a Portal */}
        <Portal>
          <CustomDialogModal />
        </Portal>
      </>
    </RootComponent>
  );
};

const styles = StyleSheet.create({
  toastContainer: {
    padding: 10,
    borderRadius: 8,
    alignSelf: 'center',
  },
  title: {
    fontWeight: 'bold',
    fontSize: 16,
    marginBottom: 4,
  },
  message: {
    fontSize: 14,
  },
  // Modal styles
  modalWrapper: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContainer: {
    backgroundColor: Colors.SECONDARY_BACKGROUND,
    borderRadius: 8,
    padding: 16,
    width: '80%',
    maxWidth: 400,
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 8,
    color: Colors.PRIMARY_TEXT,
  },
  modalMessage: {
    fontSize: 16,
    marginBottom: 16,
    color: Colors.PRIMARY_TEXT,
  },
  modalButtonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  modalCancelButton: {
    backgroundColor: Colors.DIVIDER,
    padding: 12,
    borderRadius: 8,
    flex: 1,
    marginRight: 8,
    alignItems: 'center',
  },
  modalConfirmButton: {
    backgroundColor: Colors.BUTTON,
    padding: 12,
    borderRadius: 8,
    flex: 1,
    marginLeft: 8,
    alignItems: 'center',
  },
  modalButtonText: {
    color: Colors.PRIMARY_TEXT,
    fontWeight: 'bold',
    fontSize: 14,
  },
});

export default CustomAlertRoot;
