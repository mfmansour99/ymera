import { ReactNode } from 'react';
import { NavBar } from './NavBar';

interface DashboardLayoutProps {
  children: ReactNode;
}

export function DashboardLayout({ children }: DashboardLayoutProps) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-secondary/20">
      <NavBar />
      <main className="flex-1">
        {children}
      </main>
    </div>
  );
}
