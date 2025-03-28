import React, { useState, useEffect } from 'react';
import DateTimePickerModal from 'react-native-modal-datetime-picker';
import Colors from '@/constants/Colors';

interface CustomDatePickerProps {
  date: Date;
  onConfirm: (date: Date) => void;
  onCancel: () => void;
  mode?: 'date' | 'time' | 'datetime';
  minimumDate?: Date;
  visible: boolean;
}

const CustomDatePicker: React.FC<CustomDatePickerProps> = ({
  date,
  onConfirm,
  onCancel,
  mode = 'date',
  minimumDate,
  visible,
}) => {
  const [selectedDate, setSelectedDate] = useState(date);

  useEffect(() => {
    setSelectedDate(date);
  }, [date]);

  return (
    <DateTimePickerModal
      isVisible={visible}
      mode={mode}
      date={selectedDate}
      minimumDate={minimumDate}
      onConfirm={(newDate) => {
        setSelectedDate(newDate);
        onConfirm(newDate);
      }}
      onCancel={onCancel}
      // Optionally, you can customize other props here for styling if needed.
    />
  );
};

export default CustomDatePicker;
