import React from 'react';
import {
  SafeAreaView,
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
} from 'react-native';
import { showAlert, showDialog } from '@/components/CustomAlert';
import Colors from '@/constants/Colors';

const Explore = () => {
  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>
        <Text style={styles.title}>Explore Alerts Demo</Text>
        <Text style={styles.description}>
          Press the buttons below to see different types of alerts.
        </Text>
        <TouchableOpacity
          style={styles.button}
          onPress={() =>
            showAlert({
              message: 'This is an Info alert!',
              type: 'info',
              title: 'Info Alert',
            })
          }
        >
          <Text style={styles.buttonText}>Show Info Alert</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.button}
          onPress={() =>
            showAlert({
              message: 'This is a Success alert!',
              type: 'success',
              title: 'Success Alert',
            })
          }
        >
          <Text style={styles.buttonText}>Show Success Alert</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.button}
          onPress={() =>
            showAlert({
              message: 'This is a Warning alert!',
              type: 'warning',
              title: 'Warning Alert',
            })
          }
        >
          <Text style={styles.buttonText}>Show Warning Alert</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.button}
          onPress={() =>
            showAlert({
              message: 'This is an Error alert!',
              type: 'error',
              title: 'Error Alert',
            })
          }
        >
          <Text style={styles.buttonText}>Show Error Alert</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.button}
          onPress={() =>
            showDialog({
              message: 'Are you sure you want to delete this item?',
              type: 'warning',
              title: 'Confirm Delete',
              cancelText: 'No',
              confirmText: 'Yes',
              onCancelPressed: () => console.log('Delete cancelled'),
              onConfirmPressed: () => console.log('Item deleted'),
            })
          }
        >
          <Text style={styles.buttonText}>Show Confirmation Dialog</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.PRIMARY_BACKGROUND,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
  },
  content: {
    backgroundColor: Colors.SECONDARY_BACKGROUND,
    borderRadius: 8,
    padding: 24,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOpacity: 0.3,
    shadowRadius: 10,
    elevation: 5,
    width: '100%',
  },
  title: {
    fontSize: 28,
    color: Colors.PRIMARY_TEXT,
    fontWeight: 'bold',
    marginBottom: 16,
  },
  description: {
    fontSize: 16,
    color: Colors.SECONDARY_TEXT,
    textAlign: 'center',
    marginBottom: 16,
  },
  button: {
    backgroundColor: Colors.BUTTON,
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 6,
    marginVertical: 6,
    width: '80%',
  },
  buttonText: {
    color: Colors.PRIMARY_TEXT,
    fontSize: 16,
    textAlign: 'center',
  },
});

export default Explore;
