import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { useGetTasks } from '@/api/task/useTask';

export type Task = {
    id: string;
    title: string;
    description?: string;
    status: string;
    createdAt: string;
    priority?: number;
    dueDate?: string;
    completedAt?: string | null;
    isArchived?: boolean;
    progress?: number;
};

type TasksContextType = {
    tasks: Task[];
    updateTask: (id: string, updatedTask: Partial<Task>) => void;
    deleteTask: (id: string) => void;
};

const TasksContext = createContext<TasksContextType | undefined>(undefined);

export const TasksProvider = ({ children }: { children: ReactNode }) => {
    const { data } = useGetTasks();
    const [tasks, setTasks] = useState<Task[]>([]);

    useEffect(() => {
        if (data?.tasks) {
            setTasks(data.tasks);
        }
    }, [data]);

    const updateTask = (id: string, updatedTask: Partial<Task>) => {
        setTasks(prev =>
            prev.map(task => (task.id === id ? { ...task, ...updatedTask } : task))
        );
    };

    const deleteTask = (id: string) => {
        setTasks(prev => prev.filter(task => task.id !== id));
    };

    return (
        <TasksContext.Provider value={{ tasks, updateTask, deleteTask }}>
            {children}
        </TasksContext.Provider>
    );
};

export const useTasks = () => {
    const context = useContext(TasksContext);
    if (!context) {
        throw new Error('useTasks must be used within a TasksProvider');
    }
    return context;
};
