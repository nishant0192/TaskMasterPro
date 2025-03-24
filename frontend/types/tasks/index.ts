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