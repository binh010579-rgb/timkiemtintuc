interface LogoProps {
  /** Kích thước icon logo (px). Mặc định 36. */
  size?: number
  /** Có hiển thị tên thương hiệu bên cạnh icon hay không. Mặc định true. */
  showText?: boolean
  className?: string
}

/**
 * Logo thương hiệu UTH: icon chữ "UTH" nền trắng, viền/chữ xanh dương,
 * đi kèm tên sản phẩm. Dùng chung cho Header và Footer.
 */
export default function Logo({ size = 36, showText = true, className = '' }: LogoProps) {
  return (
    <span className={`inline-flex items-center gap-2 ${className}`}>
      <img
        src="/uth-logo.svg"
        alt="UTH logo"
        width={size}
        height={size}
        className="rounded-lg shadow-sm"
      />
      {showText && (
        <span className="text-lg font-bold tracking-tight text-blue-600 dark:text-blue-400">
          UTH{' '}
          <span className="font-medium text-slate-600 dark:text-slate-300">News Search</span>
        </span>
      )}
    </span>
  )
}
