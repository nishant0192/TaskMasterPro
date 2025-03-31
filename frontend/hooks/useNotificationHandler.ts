import { useEffect } from 'react';
import * as Notifications from 'expo-notifications';
import { showAlert } from '@/components/CustomAlert';

function useNotificationHandler() {
    useEffect(() => {
        const subscription = Notifications.addNotificationReceivedListener(notification => {
            const { title, body } = notification.request.content;
            // Use custom toast to show the notification details in the foreground.
            showAlert({
                message: body || 'You have a new notification',
                type: 'info',
                title: title || 'Notification',
            });
            console.log('Notification received in foreground:', notification);
        });

        return () => {
            subscription.remove();
        };
    }, []);
}

export default useNotificationHandler;
