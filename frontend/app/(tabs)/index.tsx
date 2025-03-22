import { router } from 'expo-router';
import { Image, StyleSheet, Platform, View, Button } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

export default function HomeScreen() {
  return (
    <SafeAreaView>
      <View className='bg-white'>
        <Button
          title='Click'
          onPress={() => router.push('/auth/login')}
        />
      </View>
    </SafeAreaView>
  );
}

