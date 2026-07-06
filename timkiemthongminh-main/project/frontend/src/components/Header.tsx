import { Link } from 'react-router-dom'
import Logo from './Logo'

/**
 * Thanh header chung cho toàn bộ trang: hiển thị logo thương hiệu UTH,
 * bấm vào để quay về trang chủ. Sticky trên cùng khi cuộn trang.
 */
export default function Header() {
  return (
    <header className="sticky top-0 z-30 border-b border-slate-100 bg-white/80 backdrop-blur-md dark:border-slate-800 dark:bg-slate-900/80">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
        <Link to="/" className="transition-opacity hover:opacity-80">
          <Logo />
        </Link>
      </div>
    </header>
  )
}
