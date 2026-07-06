import { Link } from 'react-router-dom'

/**
 * Header dùng chung cho toàn bộ trang.
 * Logo là thiết kế chữ (wordmark) lấy cảm hứng từ bộ nhận diện UTH
 * (xanh ngọc + đỏ + viền vàng đồng) — không sao chép file logo gốc,
 * chỉ dùng lại tông màu thương hiệu cho phù hợp đồ án.
 */
export default function Header() {
  return (
    <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/80 backdrop-blur-md dark:border-slate-800 dark:bg-slate-900/80">
      <div className="mx-auto flex max-w-6xl items-center gap-3 px-4 py-3">
        <Link to="/" className="flex shrink-0 items-center gap-2.5">
          {/* Badge chữ UTH: nền gradient xanh ngọc, viền vàng đồng */}
          <span
            className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-teal-500 to-teal-700 text-[13px] font-black tracking-tight text-white shadow-sm ring-2 ring-amber-300/80 dark:ring-amber-400/60"
          >
            UTH
          </span>

          <span className="flex flex-col leading-none">
            <span className="text-sm font-bold text-red-600 dark:text-red-500">
              UTH News AI
            </span>
            <span className="hidden text-[11px] font-medium text-slate-500 sm:block dark:text-slate-400">
              University of Transport HCMC
            </span>
          </span>
        </Link>
      </div>
    </header>
  )
}
