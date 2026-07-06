import Logo from './Logo'

/**
 * Footer dùng chung cho Home Page (và có thể tái sử dụng ở các trang khác).
 */
export default function Footer() {
  return (
    <footer className="border-t border-slate-200 bg-slate-50 dark:border-slate-800 dark:bg-slate-900">
      <div className="mx-auto max-w-6xl px-4 py-10 text-sm text-slate-500 dark:text-slate-400">
        <div className="flex flex-col items-center justify-between gap-4 sm:flex-row">
          <Logo size={28} />
          <p>© {new Date().getFullYear()} UTH News Search. Mọi quyền được bảo lưu.</p>
        </div>
      </div>
    </footer>
  )
}
