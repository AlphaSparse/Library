#ifndef HIP_NPP_H
#define HIP_NPP_H

/** @} npp_basic_types. */

#define NPP_MIN_8U      ( 0 )                        
/**<  Minimum 8-bit unsigned integer */
#define NPP_MAX_8U      ( 255 )
/**<  Maximum 8-bit unsigned integer */
#define NPP_MIN_16U     ( 0 )
/**<  Minimum 16-bit unsigned integer */
#define NPP_MAX_16U     ( 65535 )
/**<  Maximum 16-bit unsigned integer */
#define NPP_MIN_32U     ( 0 )
/**<  Minimum 32-bit unsigned integer */
#define NPP_MAX_32U     ( 4294967295U )
/**<  Maximum 32-bit unsigned integer */
#define NPP_MIN_64U     ( 0 )
/**<  Minimum 64-bit unsigned integer */
#define NPP_MAX_64U     ( 18446744073709551615ULL )  
/**<  Maximum 64-bit unsigned integer */

#define NPP_MIN_8S      (-127 - 1 )                  
/**<  Minimum 8-bit signed integer */
#define NPP_MAX_8S      ( 127 )                      
/**<  Maximum 8-bit signed integer */
#define NPP_MIN_16S     (-32767 - 1 )
/**<  Minimum 16-bit signed integer */
#define NPP_MAX_16S     ( 32767 )
/**<  Maximum 16-bit signed integer */
#define NPP_MIN_32S     (-2147483647 - 1 )           
/**<  Minimum 32-bit signed integer */
#define NPP_MAX_32S     ( 2147483647 )
/**<  Maximum 32-bit signed integer */
#define NPP_MAX_64S     ( 9223372036854775807LL )    
/**<  Minimum 64-bit signed integer */
#define NPP_MIN_64S     (-9223372036854775807LL - 1)
/**<  Minimum 64-bit signed integer */

#define NPP_MINABS_32F  ( 1.175494351e-38f )         
/**<  Smallest positive 32-bit floating point value */
#define NPP_MAXABS_32F  ( 3.402823466e+38f )         
/**<  Largest  positive 32-bit floating point value */
#define NPP_MINABS_64F  ( 2.2250738585072014e-308 )  
/**<  Smallest positive 64-bit floating point value */
#define NPP_MAXABS_64F  ( 1.7976931348623158e+308 )  
/**<  Largest  positive 64-bit floating point value */
#endif