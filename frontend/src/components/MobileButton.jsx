import React from 'react';

/**
 * Mobile-optimized Button component with touch feedback
 * Provides larger touch targets, haptic feedback styling, and accessibility features
 */
const MobileButton = ({ 
  children, 
  onClick, 
  variant = 'primary', 
  size = 'md',
  fullWidth = false,
  disabled = false,
  type = 'button',
  className = '',
  icon: Icon,
  iconPosition = 'left'
}) => {
  const baseClasses = 'inline-flex items-center justify-center gap-2 font-semibold rounded-xl transition-all active:scale-95 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed touch-target';
  
  const variantClasses = {
    primary: 'bg-brand text-white hover:bg-brand-hover focus:ring-brand-500 active-scale ripple shadow-md',
    secondary: 'bg-white text-brand-700 border-2 border-brand hover:bg-brand-50 focus:ring-brand-500 active-scale',
    outline: 'bg-transparent border-2 border-brand-500 text-brand-700 hover:bg-brand-50 focus:ring-brand-500 active-scale',
    ghost: 'bg-transparent text-brand-700 hover:bg-brand-50 focus:ring-brand-500 active-scale',
    danger: 'bg-red-600 text-white hover:bg-red-700 focus:ring-red-500 active-scale ripple shadow-md',
  };
  
  const sizeClasses = {
    sm: 'px-4 py-2 text-sm min-h-[40px]',
    md: 'px-6 py-3 text-base min-h-[48px]',
    lg: 'px-8 py-4 text-lg min-h-[56px]',
  };
  
  const widthClass = fullWidth ? 'w-full' : '';
  
  const combinedClasses = `${baseClasses} ${variantClasses[variant]} ${sizeClasses[size]} ${widthClass} ${className}`;
  
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={combinedClasses}
    >
      {Icon && iconPosition === 'left' && <Icon className="w-5 h-5" />}
      {children}
      {Icon && iconPosition === 'right' && <Icon className="w-5 h-5" />}
    </button>
  );
};

export default MobileButton;
