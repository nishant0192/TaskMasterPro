import { Task } from '@/types/tasks';
import { create } from 'zustand';

type TasksStore = {
    tasks: Task[];
    setTasks: (tasks: Task[]) => void;
    updateTask: (id: string, updatedTask: Partial<Task>) => void;
    deleteTask: (id: string) => void;
  };
  
  export const useTasksStore = create<TasksStore>((set) => ({
    tasks: [],
    setTasks: (tasks: Task[]) => set({ tasks }),
    updateTask: (id: string, updatedTask: Partial<Task>) =>
      set((state) => ({
        tasks: state.tasks.map(task =>
          task.id === id ? { ...task, ...updatedTask } : task
        ),
      })),
    deleteTask: (id: string) =>
      set((state) => ({ tasks: state.tasks.filter(task => task.id !== id) })),
  }));