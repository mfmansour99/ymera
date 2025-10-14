export const LoginSkeleton = () => (
  <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
    <div className="text-center">
      <div className="w-16 h-16 border-4 border-cyan-500/30 border-t-cyan-500 rounded-full animate-spin mx-auto mb-4" />
      <h2 className="text-xl font-semibold text-white mb-2">Loading Neural Grid</h2>
      <p className="text-slate-400">Initializing authentication system...</p>
    </div>
  </div>
);