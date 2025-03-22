import { Platform, TextStyle } from "react-native";

const textStyles: { [key: string]: TextStyle } = {
  splashHeading: {
    fontFamily: Platform.select({
      android: "Poppins_600SemiBold",
      ios: "Poppins-SemiBold",
    }) as string,
    fontSize: 48,
  },
  pageHeader: {
    fontFamily: Platform.select({
      android: "Poppins_600SemiBold",
      ios: "Poppins-SemiBold",
    }) as string,
    fontSize: 32,
  },
  pageSubheader: {
    fontFamily: Platform.select({
      android: "Poppins_400Regular",
      ios: "Poppins-Regular",
    }) as string,
    fontSize: 16,
    lineHeight: 22,
    color: "#767676",
  },
  headingBig: {
    fontFamily: Platform.select({
      android: "Poppins_600SemiBold",
      ios: "Poppins-SemiBold",
    }) as string,
    fontSize: 32,
    lineHeight: 40,
  },
  headingMedium: {
    fontFamily: Platform.select({
      android: "Poppins_600SemiBold",
      ios: "Poppins-SemiBold",
    }) as string,
    fontSize: 16,
    lineHeight: 24,
  },
  headingSmall: {
    fontFamily: Platform.select({
      android: "Poppins_600SemiBold",
      ios: "Poppins-SemiBold",
    }) as string,
    fontSize: 12,
    lineHeight: 18,
  },
  buttonLarge: {
    fontFamily: Platform.select({
      android: "Poppins_700Bold",
      ios: "Poppins-Bold",
    }) as string,
    fontSize: 24,
  },
  buttonSmall: {
    fontFamily: Platform.select({
      android: "Poppins_700Bold",
      ios: "Poppins-Bold",
    }) as string,
    fontSize: 16,
  },
  consentText: {
    fontFamily: Platform.select({
      android: "Poppins_400Regular",
      ios: "Poppins-Regular",
    }) as string,
    fontSize: 12,
  },
  linkMediumHeading: {
    fontFamily: Platform.select({
      android: "Poppins_600SemiBold",
      ios: "Poppins-SemiBold",
    }) as string,
    fontSize: 16,
    textDecorationLine: "underline",
  },
  appbarHeading: {
    fontFamily: Platform.select({
      android: "Poppins_600SemiBold",
      ios: "Poppins-SemiBold",
    }) as string,
    fontSize: 16,
  },
  appbarButton: {
    fontFamily: Platform.select({
      android: "Poppins_700Bold",
      ios: "Poppins-Bold",
    }) as string,
    fontSize: 16,
    textDecorationLine: "underline",
  },
  extraLarge: {
    fontFamily: Platform.select({
      android: "Poppins_600SemiBold",
      ios: "Poppins-SemiBold",
    }) as string,
    fontSize: 48,
    lineHeight: 60,
  },
};

export default textStyles;
